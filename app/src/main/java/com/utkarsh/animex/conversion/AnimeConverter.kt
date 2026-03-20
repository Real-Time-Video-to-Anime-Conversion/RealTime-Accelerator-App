package com.utkarsh.animex.conversion

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream
import androidx.core.graphics.createBitmap
import androidx.core.graphics.set
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AnimeConverter(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var outputBuffer: ByteBuffer? = null
    var lastInferenceTime: Long = 0
        private set

    companion object {
        const val INPUT_SIZE = 512
    }

    init {
        loadModel()
        printModelInfo()
    }

    private fun loadModel() {
        val modelBuffer = loadModelFile("AnimeGANv3_Hayao_36_QUANTIZED_int8.tflite")
        interpreter = Interpreter(modelBuffer)
    }

    private fun printModelInfo() {
        val inputTensor = interpreter?.getInputTensor(0)
        val outputTensor = interpreter?.getOutputTensor(0)

        println("=== MODEL INPUT INFO ===")
        println("Type: ${inputTensor?.dataType()}")
        println("Shape: ${inputTensor?.shape()?.joinToString()}")

        println("=== MODEL OUTPUT INFO ===")
        println("Type: ${outputTensor?.dataType()}")
        println("Shape: ${outputTensor?.shape()?.joinToString()}")
    }

    fun convert(inputImage: TensorImage): Bitmap {
        val interpreter = interpreter ?: throw IllegalStateException("Model not loaded")

        val inputBuffer = inputImage.buffer

        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()

        val size = outputShape[1] * outputShape[2] * outputShape[3]

        if (outputBuffer == null) {
            outputBuffer = ByteBuffer.allocateDirect(size)
                .order(ByteOrder.nativeOrder())
        }

        val buffer = outputBuffer!!
        buffer.rewind()

        val start = System.currentTimeMillis()
        interpreter.run(inputBuffer, buffer)
        lastInferenceTime = System.currentTimeMillis() - start
        println("Inference: ${lastInferenceTime}ms")

        buffer.rewind()
        return uint8BufferToBitmap(buffer, outputShape[2], outputShape[1])
    }

    private fun uint8BufferToBitmap(
        buffer: ByteBuffer,
        width: Int,
        height: Int
    ): Bitmap {

        buffer.rewind()

        val bitmap = createBitmap(width, height, Bitmap.Config.ARGB_8888)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = buffer.get().toInt() and 0xFF
                val g = buffer.get().toInt() and 0xFF
                val b = buffer.get().toInt() and 0xFF

                bitmap[x, y] = android.graphics.Color.rgb(r, g, b)
            }
        }

        return bitmap
    }

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel

        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    fun close() {
        interpreter?.close()
    }
}
