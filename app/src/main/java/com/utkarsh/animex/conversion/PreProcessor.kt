package com.utkarsh.animex.conversion

import android.graphics.Bitmap
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.support.image.TensorImage
import androidx.core.graphics.scale
import androidx.core.graphics.createBitmap

class PreProcessor {

    // Model constants
    companion object {
        const val MODEL_INPUT_SIZE = 512  // 512x512
    }

    // STEP 2: Preprocess camera image for the model
    fun preprocess(imageProxy: ImageProxy): TensorImage {
        // STEP 2a: Convert YUV to RGB Bitmap
        val originalBitmap = yuv420888ToRgbBitmap(imageProxy)

        // STEP 2b: Resize to match model input (512x512)
        val resizedBitmap = resizeBitmap(originalBitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        // STEP 2c: Convert Bitmap to TensorImage
        val tensorImage = TensorImage.fromBitmap(resizedBitmap)

        return tensorImage
    }

    // Resize bitmap to exact model dimensions
    private fun resizeBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {

        val width = bitmap.width
        val height = bitmap.height

        val cropSize = minOf(width, height)

        val x = (width - cropSize) / 2
        val y = (height - cropSize) / 2

        val cropped = Bitmap.createBitmap(bitmap, x, y, cropSize, cropSize)

        return cropped.scale(targetWidth, targetHeight)
    }

    // YUV to RGB conversion (simplified version)
    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    private fun yuv420888ToRgbBitmap(imageProxy: ImageProxy): Bitmap {
        val image = imageProxy.image ?: throw IllegalStateException("Image is null")

        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val bitmap = createBitmap(width, height)

        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            val yRow = yRowStride * y
            val uvRow = uvRowStride * (y / 2)

            for (x in 0 until width) {
                val yIndex = yRow + x
                val uvIndex = uvRow + (x / 2) * uvPixelStride

                val yValue = (yBuffer.get(yIndex).toInt() and 0xFF)
                val uValue = (uBuffer.get(uvIndex).toInt() and 0xFF)
                val vValue = (vBuffer.get(uvIndex).toInt() and 0xFF)

                // Convert YUV → RGB
                val r = (yValue + 1.370705f * (vValue - 128)).toInt()
                val g = (yValue - 0.337633f * (uValue - 128) - 0.698001f * (vValue - 128)).toInt()
                val b = (yValue + 1.732446f * (uValue - 128)).toInt()

                val rClamped = r.coerceIn(0, 255)
                val gClamped = g.coerceIn(0, 255)
                val bClamped = b.coerceIn(0, 255)

                pixels[y * width + x] =
                    (0xFF shl 24) or (rClamped shl 16) or (gClamped shl 8) or bClamped
            }
        }

        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

        // Apply rotation
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        if (rotationDegrees != 0) {
            val matrix = android.graphics.Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
        }

        return bitmap
    }
}
