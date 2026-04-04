package com.utkarsh.animex.conversion

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream
import androidx.core.graphics.createBitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AnimeConverter(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var outputBuffer: ByteBuffer? = null
    private val nativeProcessor = NativeProcessor()
    
    // Buffers for Sobel logic
    private var grayBuffer: ByteBuffer? = null
    private var edgeBuffer: ByteBuffer? = null

    companion object {
        const val INPUT_SIZE = 512
    }

    init {
        loadModel()
    }

    private fun loadModel() {
        val modelBuffer = loadModelFile("AnimeGANv3_Hayao_36_QUANTIZED_int8.tflite")
        val options = Interpreter.Options().setNumThreads(4)
        interpreter = Interpreter(modelBuffer, options)
    }

    fun convert(inputImage: TensorImage, originalBitmap: Bitmap, useSIMD: Boolean): Bitmap {
        val interpreter = interpreter ?: throw IllegalStateException("Model not loaded")

        val inputBuffer = inputImage.buffer
        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape() // [1, 512, 512, 3]

        val height = outputShape[1]
        val width = outputShape[2]
        val size = height * width * 3

        if (outputBuffer == null) {
            outputBuffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
            grayBuffer = ByteBuffer.allocateDirect(originalBitmap.width * originalBitmap.height).order(ByteOrder.nativeOrder())
            edgeBuffer = ByteBuffer.allocateDirect(originalBitmap.width * originalBitmap.height).order(ByteOrder.nativeOrder())
        }

        val buffer = outputBuffer!!
        buffer.rewind()

        // 1. Run ML Inference
        interpreter.run(inputBuffer, buffer)

        if (!useSIMD) {
            // Baseline slow path
            return uint8BufferToBitmap(buffer, width, height)
        }

        // 2. SIMD Path (Stage 4, 5, 7)
        val resultBitmap = createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // Resize original to match model for edge detection (Simplified for now, using ML size)
        // In a real pipeline, we'd do Sobel on full res, but let's stick to the report's flow
        val mlResBitmap = Bitmap.createScaledBitmap(originalBitmap, width, height, true)
        
        nativeProcessor.rgbToGrayscaleSIMD(mlResBitmap, grayBuffer!!, width, height)
        nativeProcessor.sobelEdgeSIMD(grayBuffer!!, edgeBuffer!!, width, height, 50)
        
        buffer.rewind()
        nativeProcessor.compositeSIMD(buffer, edgeBuffer!!, resultBitmap, width, height)

        return resultBitmap
    }

    private fun uint8BufferToBitmap(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        buffer.rewind()
        val bitmap = createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = buffer.get().toInt() and 0xFF
                val g = buffer.get().toInt() and 0xFF
                val b = buffer.get().toInt() and 0xFF
                bitmap.setPixel(x, y, android.graphics.Color.rgb(r, g, b))
            }
        }
        return bitmap
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        return inputStream.channel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    fun close() {
        interpreter?.close()
    }
}
