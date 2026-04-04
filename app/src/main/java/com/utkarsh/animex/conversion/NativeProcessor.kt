package com.utkarsh.animex.conversion

import android.graphics.Bitmap
import java.nio.ByteBuffer

class NativeProcessor {
    companion object {
        init {
            System.loadLibrary("animex")
        }
    }

    /**
     * Converts YUV_420_888 to RGBA_8888 using ARM NEON SIMD.
     */
    external fun yuvToRgbaSIMD(
        yBuffer: ByteBuffer,
        yRowStride: Int,
        uBuffer: ByteBuffer,
        uRowStride: Int,
        vBuffer: ByteBuffer,
        vRowStride: Int,
        uvPixelStride: Int,
        outBitmap: Bitmap,
        width: Int,
        height: Int
    )

    /**
     * Converts RGB Bitmap to Grayscale using ARM NEON SIMD.
     */
    external fun rgbToGrayscaleSIMD(
        bitmap: Bitmap,
        grayBuffer: ByteBuffer,
        width: Int,
        height: Int
    )

    /**
     * Applies Sobel Edge Detection using ARM NEON SIMD.
     */
    external fun sobelEdgeSIMD(
        grayBuffer: ByteBuffer,
        edgeBuffer: ByteBuffer,
        width: Int,
        height: Int,
        threshold: Int
    )

    /**
     * Composites ML Output and Edge Map using ARM NEON SIMD.
     */
    external fun compositeSIMD(
        mlBuffer: ByteBuffer,
        edgeBuffer: ByteBuffer,
        outBitmap: Bitmap,
        width: Int,
        height: Int
    )
}
