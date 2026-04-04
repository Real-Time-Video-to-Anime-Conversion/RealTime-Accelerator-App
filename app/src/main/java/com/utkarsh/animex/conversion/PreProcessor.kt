package com.utkarsh.animex.conversion

import android.graphics.Bitmap
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.support.image.TensorImage
import androidx.core.graphics.scale
import androidx.core.graphics.createBitmap

class PreProcessor {

    private val nativeProcessor = NativeProcessor()

    companion object {
        const val MODEL_INPUT_SIZE = 512
    }

    fun preprocess(imageProxy: ImageProxy, useSIMD: Boolean): Pair<TensorImage, Bitmap> {
        val originalBitmap = if (useSIMD) {
            yuvToRgbSIMD(imageProxy)
        } else {
            yuvToRgbBaseline(imageProxy)
        }

        val resizedBitmap = resizeBitmap(originalBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        return TensorImage.fromBitmap(resizedBitmap) to originalBitmap
    }

    private fun resizeBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val cropSize = minOf(width, height)
        val x = (width - cropSize) / 2
        val y = (height - cropSize) / 2
        val cropped = Bitmap.createBitmap(bitmap, x, y, cropSize, cropSize)
        return cropped.scale(targetWidth, targetHeight)
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    private fun yuvToRgbSIMD(imageProxy: ImageProxy): Bitmap {
        val image = imageProxy.image ?: throw IllegalStateException("Image is null")
        val bitmap = createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        
        nativeProcessor.yuvToRgbaSIMD(
            image.planes[0].buffer, image.planes[0].rowStride,
            image.planes[1].buffer, image.planes[1].rowStride,
            image.planes[2].buffer, image.planes[2].rowStride,
            image.planes[1].pixelStride,
            bitmap, image.width, image.height
        )
        return applyRotation(bitmap, imageProxy.imageInfo.rotationDegrees)
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    private fun yuvToRgbBaseline(imageProxy: ImageProxy): Bitmap {
        val image = imageProxy.image ?: throw IllegalStateException("Image is null")
        val width = image.width
        val height = image.height
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val yRowStride = image.planes[0].rowStride
        val uvRowStride = image.planes[1].rowStride
        val uvPixelStride = image.planes[1].pixelStride

        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            val yRow = yRowStride * y
            val uvRow = uvRowStride * (y / 2)
            for (x in 0 until width) {
                val yIndex = yRow + x
                val uvIndex = uvRow + (x / 2) * uvPixelStride
                val yVal = yBuffer.get(yIndex).toInt() and 0xFF
                val uVal = uBuffer.get(uvIndex).toInt() and 0xFF
                val vVal = vBuffer.get(uvIndex).toInt() and 0xFF

                val r = (yVal + 1.370705f * (vVal - 128)).toInt().coerceIn(0, 255)
                val g = (yVal - 0.337633f * (uVal - 128) - 0.698001f * (vValue - 128)).toInt().coerceIn(0, 255)
                val b = (yVal + 1.732446f * (uVal - 128)).toInt().coerceIn(0, 255)
                pixels[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        val bitmap = createBitmap(width, height)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return applyRotation(bitmap, imageProxy.imageInfo.rotationDegrees)
    }

    private fun applyRotation(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}
