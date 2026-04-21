package com.utkarsh.animex.conversion

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import com.utkarsh.animex.screens.InferenceMode
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import android.graphics.Bitmap.createBitmap
import kotlin.math.roundToInt

/**
 * Runs TFLite inference for one of three pipeline modes:
 *
 *  NORMAL  — CPU-only TFLite, pure-Java pre/post-processing.
 *            Model: int8 quantized  (AnimeGANv3_Hayao_36_QUANTIZED_int8.tflite)
 *            Baseline correctness reference; no NEON or GPU.
 *
 *  SIMD    — CPU TFLite with XNNPACK, NEON SIMD pre/post-processing.
 *            Model: int8 quantized  (AnimeGANv3_Hayao_36_QUANTIZED_int8.tflite)
 *            Best CPU throughput via ARM NEON intrinsics (8 px/cycle).
 *
 *  HYBRID  — GPU delegate TFLite, NEON SIMD pre/post-processing.
 *            Model: fp16 quantized  (AnimeGANv3_Hayao_36_QUANTIZED_fp16.tflite)
 *            Lowest latency on devices with a capable GPU.
 *            fp16 runs natively on mobile GPUs without dequantization overhead.
 *
 * In all modes the model expects float32 input [1, 512, 512, 3] in [-1, 1]
 * and produces float32 or uint8 output [1, 512, 512, 3], matching training:
 *   input  = pixel / 127.5 - 1.0
 *   output = (value + 1.0) * 127.5  → clamped uint8  (float output only)
 */
class TFLiteRunner(
    context: Context,
    mode: InferenceMode = InferenceMode.NORMAL
) {
    companion object {
        private const val TAG = "TFLiteRunner"

        /** Model asset selected per mode. */
        private fun modelFileFor(mode: InferenceMode): String = when (mode) {
            InferenceMode.NORMAL,
            InferenceMode.SIMD   -> "AnimeGANv3_Hayao_36_QUANTIZED_int8.tflite"
            InferenceMode.HYBRID -> "AnimeGANv3_Hayao_36_QUANTIZED_fp16.tflite"
            InferenceMode.GPU -> {
                error("TFLiteRunner must not be constructed for GPU mode")
            }
        }
    }

    // ── Interpreter & delegate ────────────────────────────────────────────────

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    // ── Output tensor reuse ───────────────────────────────────────────────────

    /** Reused direct ByteBuffer for model output (avoids per-frame allocation). */
    private var outputBuffer: ByteBuffer? = null

    /**
     * Double-buffered output bitmaps: while one is being displayed the other
     * can be written into, preventing tearning without extra copies.
     */
    private val outputBitmaps = arrayOfNulls<Bitmap>(2)
    private var outputBitmapIndex = 0

    /** Reused pixel int[] for NEON → setPixels path. */
    private var outputPixels: IntArray = IntArray(0)

    /** Whether NEON post-processing is active (SIMD / HYBRID). */
    private val useNeon = mode == InferenceMode.SIMD || mode == InferenceMode.HYBRID
    private var outputScale: Float = 1f
    private var outputZeroPoint: Int = 0


    // ── Init ──────────────────────────────────────────────────────────────────

    init {
        val modelFile = modelFileFor(mode)
        try {
            val options = Interpreter.Options().apply {
                when (mode) {
                    InferenceMode.NORMAL -> {
                        setNumThreads(4)
                        setUseXNNPACK(true)
                    }
                    InferenceMode.SIMD -> {
                        setNumThreads(4)
                        setUseXNNPACK(true)
                    }
                    InferenceMode.HYBRID,
                    InferenceMode.GPU     -> { setNumThreads(1); gpuDelegate = GpuDelegate(
                        GpuDelegateFactory.Options().apply {
                            setPrecisionLossAllowed(true)
                        }
                    ); addDelegate(gpuDelegate!!) }
                }
            }
            interpreter = Interpreter(loadModelFile(context, modelFile), options)
            interpreter?.getOutputTensor(0)?.quantizationParams()?.let {
                outputScale = it.scale
                outputZeroPoint = it.zeroPoint
            }
            printModelInfo(mode)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model $modelFile in $mode mode", e)
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Runs inference on [inputBuffer] (float32 direct ByteBuffer, [1,512,512,3]).
     * Returns an ARGB [Bitmap] or null on error.
     * Thread-safe; serialized by [Synchronized].
     */
    @Synchronized
    fun run(inputBuffer: ByteBuffer): Bitmap? {
        val interp = interpreter ?: return null

        return try {
            require(inputBuffer.isDirect) { "inputBuffer must be a direct ByteBuffer" }

            val outputTensor = interp.getOutputTensor(0)
            val shape        = outputTensor.shape()   // [1, H, W, C]

            if (shape.size < 4 || shape[1] <= 0 || shape[2] <= 0) {
                Log.e(TAG, "Unexpected output shape: ${shape.joinToString()}")
                return null
            }

            val isFloat         = outputTensor.dataType() == DataType.FLOAT32
            val bytesPerElement = if (isFloat) Float.SIZE_BYTES else 1
            val totalBytes      = shape[1] * shape[2] * shape[3] * bytesPerElement

            val buffer = getOrReallocateOutputBuffer(totalBytes)
            buffer.rewind()
            interp.run(inputBuffer, buffer)
            buffer.rewind()

            val width  = shape[2]
            val height = shape[1]

            if (useNeon) {
                // SIMD / HYBRID: NEON post-processing
                if (isFloat) floatTensorToBitmapNeon(buffer, width, height)
                else         uint8TensorToBitmapNeon(buffer, width, height)
            } else {
                // NORMAL: pure-Java post-processing
                if (isFloat) floatTensorToBitmapJava(buffer, width, height)
                else         uint8TensorToBitmapJava(buffer, width, height)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            null
        }
    }

    @Synchronized
    fun close() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter  = null
        gpuDelegate  = null
        outputBuffer = null
        outputBitmaps.fill(null)
        outputPixels = IntArray(0)
    }

    // ── NEON post-processing (SIMD / HYBRID) ──────────────────────────────────

    private fun floatTensorToBitmapNeon(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        val pixels = getOrResizePixelBuffer(width * height)
        NeonProcessor.floatTensorToArgb(buffer, width, height, pixels)
        val bitmap = getOrRecreateBitmap(width, height)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun uint8TensorToBitmapNeon(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        val pixels = getOrResizePixelBuffer(width * height)
        NeonProcessor.uint8QuantizedTensorToArgb(
            tensorBuf = buffer,
            width = width,
            height = height,
            scale = outputScale,
            zeroPoint = outputZeroPoint,
            outPixels = pixels
        )
        val bitmap = getOrRecreateBitmap(width, height)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }


    // ── Pure-Java post-processing (NORMAL) ────────────────────────────────────

    /**
     * Denormalizes float32 output [-1,1] → uint8 [0,255]:
     *   pixel = (value + 1.0) * 127.5   (mirrors training normalization exactly)
     */
    private fun floatTensorToBitmapJava(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        val pixels = getOrResizePixelBuffer(width * height)
        for (i in pixels.indices) {
            val r = denorm(buffer.float)
            val g = denorm(buffer.float)
            val b = denorm(buffer.float)
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bitmap = getOrRecreateBitmap(width, height)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun uint8TensorToBitmapJava(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        val pixels = getOrResizePixelBuffer(width * height)
        for (i in pixels.indices) {
            val r = buffer.get().toInt() and 0xFF
            val g = buffer.get().toInt() and 0xFF
            val b = buffer.get().toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bitmap = getOrRecreateBitmap(width, height)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    /** Denormalize a single float32 value from [-1,1] → clamped [0,255]. */
    private fun denorm(v: Float): Int =
        ((v + 1f) * 127.5f).toInt().coerceIn(0, 255)

    // ── Buffer helpers ────────────────────────────────────────────────────────

    private fun getOrReallocateOutputBuffer(totalBytes: Int): ByteBuffer {
        val existing = outputBuffer
        if (existing != null && existing.capacity() == totalBytes) return existing
        return ByteBuffer.allocateDirect(totalBytes)
            .order(ByteOrder.nativeOrder())
            .also { outputBuffer = it }
    }

    private fun getOrResizePixelBuffer(size: Int): IntArray {
        if (outputPixels.size != size) outputPixels = IntArray(size)
        return outputPixels
    }

    private fun getOrRecreateBitmap(width: Int, height: Int): Bitmap {
        outputBitmapIndex = (outputBitmapIndex + 1) % outputBitmaps.size
        val existing = outputBitmaps[outputBitmapIndex]
        if (existing != null && !existing.isRecycled
            && existing.width == width && existing.height == height
        ) return existing
        return createBitmap(width, height, Bitmap.Config.ARGB_8888)
            .also { outputBitmaps[outputBitmapIndex] = it }
    }

    // ── Asset loader ──────────────────────────────────────────────────────────

    private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
        val fd = context.assets.openFd(modelFile)
        return FileInputStream(fd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            fd.startOffset,
            fd.declaredLength
        )
    }

    private fun printModelInfo(mode: InferenceMode) {
        val interp = interpreter ?: return

        val input = interp.getInputTensor(0)
        val output = interp.getOutputTensor(0)

        val inQ = input.quantizationParams()
        val outQ = output.quantizationParams()

        Log.d(TAG, "[$mode] Input dtype=${input.dataType()}")
        Log.d(TAG, "[$mode] Input shape=${input.shape().joinToString()}")
        Log.d(TAG, "[$mode] Input quant: scale=${inQ.scale}, zeroPoint=${inQ.zeroPoint}")

        Log.d(TAG, "[$mode] Output dtype=${output.dataType()}")
        Log.d(TAG, "[$mode] Output shape=${output.shape().joinToString()}")
        Log.d(TAG, "[$mode] Output quant: scale=${outQ.scale}, zeroPoint=${outQ.zeroPoint}")
    }

}