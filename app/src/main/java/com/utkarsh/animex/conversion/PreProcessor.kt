package com.utkarsh.animex.conversion

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.graphics.Bitmap.createBitmap
import androidx.core.graphics.withMatrix

/**
 * Preprocesses camera frames into a model-ready float32 ByteBuffer
 * shaped [1, 512, 512, 3], normalized to [-1, 1] to match training:
 *   pixel_float = pixel_uint8 / 127.5 - 1.0
 *
 * Two paths are available:
 *   • useNeon = false  →  pure-Java/Kotlin (NORMAL mode)
 *   • useNeon = true   →  NEON SIMD YUV→ARGB then Java normalization (SIMD / HYBRID mode)
 *
 * Output is always a direct ByteBuffer so TFLite can DMA it without a copy.
 */
class PreProcessor(
    private val useNeon: Boolean = false,
    private val quantizedInput: Boolean = false
) {

    companion object {
        const val MODEL_INPUT_SIZE = 512
        private const val CHANNELS = 3

        /** Total bytes for one float32 [1, H, W, 3] input tensor. */
        const val INPUT_TENSOR_BYTES_FLOAT: Int =
            MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * CHANNELS * Float.SIZE_BYTES

        const val INPUT_TENSOR_BYTES_UINT8: Int =
            MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * CHANNELS

    }

    // ── Reusable allocations (avoid per-frame GC) ─────────────────────────────

    // Full-resolution RGB bitmap (lazy, reallocated only on resolution change)
    private var rgbBitmap: Bitmap? = null

    // 512×512 scratch bitmap used for crop-and-scale output
    private val scaledBitmap: Bitmap = createBitmap(
        MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888
    )
    private val scaledCanvas  = Canvas(scaledBitmap)
    private val scalePaint    = Paint(Paint.FILTER_BITMAP_FLAG)
    private val rotationMatrix = Matrix()
    private val srcRect       = Rect()
    private val dstRectF      = RectF(0f, 0f,
        MODEL_INPUT_SIZE.toFloat(), MODEL_INPUT_SIZE.toFloat())

    // NEON path: pixel buffer reused across frames
    private var neonPixelBuffer: IntArray = IntArray(0)

    // Normalized pixel buffer (ARGB int → packed RGB floats)
    private val scaledPixels = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)

    // Single pre-allocated output tensor buffer (direct, native-order)
    val outputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(
            if (quantizedInput) INPUT_TENSOR_BYTES_UINT8 else INPUT_TENSOR_BYTES_FLOAT
        )
        .order(ByteOrder.nativeOrder())
    @Volatile
    private var isReleased = false


    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Converts [imageProxy] → model-ready float32 ByteBuffer.
     * Caller must NOT close [imageProxy] before this returns.
     * The returned buffer is the same object as [outputBuffer]; it is rewound
     * and ready to pass straight to TFLite.
     */
    @Synchronized
    @OptIn(ExperimentalGetImage::class)
    fun preprocess(imageProxy: ImageProxy): ByteBuffer {
        check(!isReleased) { "PreProcessor used after release" }

        if (useNeon) yuvToRgbNeon(imageProxy) else yuvToRgbJava(imageProxy)
        cropAndScale(imageProxy.imageInfo.rotationDegrees)

        if (quantizedInput) {
            packToUint8Buffer()
        } else {
            normalizeToFloatBuffer()
        }

        return outputBuffer
    }

    @Synchronized
    fun release() {
        if (isReleased) return
        isReleased = true

        rgbBitmap?.recycle()
        rgbBitmap = null
        neonPixelBuffer = IntArray(0)
    }


    // ── Step 1a: NEON SIMD YUV → RGB bitmap ──────────────────────────────────

    @OptIn(ExperimentalGetImage::class)
    private fun yuvToRgbNeon(imageProxy: ImageProxy) {
        val image = imageProxy.image
            ?: throw IllegalStateException("ImageProxy.image is null")

        val w = image.width
        val h = image.height
        ensureRgbBitmap(w, h)

        // Grow the pixel buffer only when resolution changes
        val needed = w * h
        if (neonPixelBuffer.size < needed) neonPixelBuffer = IntArray(needed)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        NeonProcessor.yuvToArgb(
            yBuf          = yPlane.buffer,
            uBuf          = uPlane.buffer,
            vBuf          = vPlane.buffer,
            yRowStride    = yPlane.rowStride,
            uvRowStride   = uPlane.rowStride,
            uvPixelStride = uPlane.pixelStride,
            width         = w,
            height        = h,
            outPixels     = neonPixelBuffer
        )
        rgbBitmap!!.setPixels(neonPixelBuffer, 0, w, 0, 0, w, h)
    }

    // ── Step 1b: Pure-Java YUV → RGB bitmap (NORMAL mode) ────────────────────

    @OptIn(ExperimentalGetImage::class)
    private fun yuvToRgbJava(imageProxy: ImageProxy) {
        val image = imageProxy.image
            ?: throw IllegalStateException("ImageProxy.image is null")

        val w = image.width
        val h = image.height
        ensureRgbBitmap(w, h)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuf          = yPlane.buffer
        val uBuf          = uPlane.buffer
        val vBuf          = vPlane.buffer
        val yRowStride    = yPlane.rowStride
        val uvRowStride   = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val needed = w * h
        if (neonPixelBuffer.size < needed) neonPixelBuffer = IntArray(needed)

        for (row in 0 until h) {
            for (col in 0 until w) {
                val yIdx  = row * yRowStride + col
                val uvIdx = (row shr 1) * uvRowStride + (col shr 1) * uvPixelStride

                val yVal = (yBuf.get(yIdx).toInt() and 0xFF)
                val uVal = (uBuf.get(uvIdx).toInt() and 0xFF) - 128
                val vVal = (vBuf.get(uvIdx).toInt() and 0xFF) - 128

                var r = yVal + (1403 * vVal shr 10)
                var g = yVal - (346  * uVal shr 10) - (715 * vVal shr 10)
                var b = yVal + (1774 * uVal shr 10)

                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)

                neonPixelBuffer[row * w + col] =
                    (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        rgbBitmap!!.setPixels(neonPixelBuffer, 0, w, 0, 0, w, h)
    }

    // ── Step 2: center-crop + scale ───────────────────────────────────────────

    private fun cropAndScale(rotationDegrees: Int) {
        val src = rgbBitmap ?: return
        val cropSize = minOf(src.width, src.height)
        val cropX = (src.width  - cropSize) / 2
        val cropY = (src.height - cropSize) / 2
        srcRect.set(cropX, cropY, cropX + cropSize, cropY + cropSize)

        scaledCanvas.drawColor(android.graphics.Color.BLACK)

        if (rotationDegrees == 0) {
            scaledCanvas.drawBitmap(src, srcRect, dstRectF, scalePaint)
            return
        }

        rotationMatrix.reset()
        rotationMatrix.postRotate(
            rotationDegrees.toFloat(),
            MODEL_INPUT_SIZE / 2f,
            MODEL_INPUT_SIZE / 2f
        )
        scaledCanvas.withMatrix(rotationMatrix) {
            drawBitmap(src, srcRect, dstRectF, scalePaint)
        }
    }

    // ── Step 3: ARGB bitmap → float32 [-1,1] direct ByteBuffer ───────────────

    /**
     * Reads the 512×512 ARGB bitmap row-by-row and writes interleaved
     * R, G, B float32 values normalized with:  f = pixel / 127.5 - 1.0
     *
     * This exactly mirrors the training pipeline:
     *   processing_image = image / 127.5 - 1.0
     */
    private fun normalizeToFloatBuffer() {
        scaledBitmap.getPixels(
            scaledPixels, 0, MODEL_INPUT_SIZE,
            0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
        )

        outputBuffer.rewind()
        val inv = 1f / 127.5f

        for (pixel in scaledPixels) {
            val r = ((pixel shr 16) and 0xFF).toFloat() * inv - 1f
            val g = ((pixel shr 8) and 0xFF).toFloat() * inv - 1f
            val b = (pixel and 0xFF).toFloat() * inv - 1f
            outputBuffer.putFloat(r)
            outputBuffer.putFloat(g)
            outputBuffer.putFloat(b)
        }

        outputBuffer.rewind()
    }

    private fun packToUint8Buffer() {
        scaledBitmap.getPixels(
            scaledPixels, 0, MODEL_INPUT_SIZE,
            0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
        )

        outputBuffer.rewind()

        for (pixel in scaledPixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF

            outputBuffer.put(r.toByte())
            outputBuffer.put(g.toByte())
            outputBuffer.put(b.toByte())
        }

        outputBuffer.rewind()
    }



    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun ensureRgbBitmap(w: Int, h: Int) {
        val cur = rgbBitmap
        if (cur == null || cur.width != w || cur.height != h) {
            cur?.recycle()
            rgbBitmap = createBitmap(w, h, Bitmap.Config.ARGB_8888)
        }
    }
}