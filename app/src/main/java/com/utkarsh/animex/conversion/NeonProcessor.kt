package com.utkarsh.animex.conversion

import java.nio.ByteBuffer

/**
 * JNI bridge to the NEON SIMD native library.
 *
 * All functions operate on direct ByteBuffers (zero-copy from JNI side)
 * and write results into pre-allocated IntArrays to avoid per-frame allocation.
 *
 * SAFETY CONTRACT (enforced here before every JNI call):
 *   • All ByteBuffers must be direct (GetDirectBufferAddress returns non-null).
 *   • width and height must both be > 0.
 *   • outPixels.size must equal width * height exactly.
 * Violating any of these causes a SIGSEGV in native code; the guards below
 * throw IllegalArgumentException instead so the caller can handle gracefully.
 */
object NeonProcessor {

    init {
        System.loadLibrary("animex_neon")
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private fun requireDirect(buf: ByteBuffer, name: String) {
        require(buf.isDirect) { "$name must be a direct ByteBuffer" }
    }

    private fun requireDimensions(width: Int, height: Int) {
        require(width > 0 && height > 0) {
            "Invalid dimensions: ${width}x${height} — both must be > 0"
        }
    }

    private fun requireOutputSize(outPixels: IntArray, width: Int, height: Int) {
        val expected = width * height
        require(outPixels.size >= expected) {
            "outPixels too small: got ${outPixels.size}, need $expected ($width × $height)"
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Converts a YUV_420_888 frame to ARGB pixels using NEON SIMD.
     * Processes 8 pixels per cycle via ARM NEON intrinsics.
     *
     * @param yBuf          Y plane direct ByteBuffer
     * @param uBuf          U plane direct ByteBuffer
     * @param vBuf          V plane direct ByteBuffer
     * @param yRowStride    Y plane row stride in bytes
     * @param uvRowStride   UV plane row stride in bytes
     * @param uvPixelStride UV plane pixel stride (1=planar, 2=interleaved)
     * @param width         frame width in pixels
     * @param height        frame height in pixels
     * @param outPixels     pre-allocated ARGB output array (size >= width * height)
     * @throws IllegalArgumentException if any buffer is non-direct, dimensions
     *         are invalid, or outPixels is too small.
     */
    fun yuvToArgb(
        yBuf: ByteBuffer,
        uBuf: ByteBuffer,
        vBuf: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        width: Int,
        height: Int,
        outPixels: IntArray
    ) {
        requireDirect(yBuf, "yBuf")
        requireDirect(uBuf, "uBuf")
        requireDirect(vBuf, "vBuf")
        requireDimensions(width, height)
        requireOutputSize(outPixels, width, height)
        yuvToArgbNative(yBuf, uBuf, vBuf, yRowStride, uvRowStride, uvPixelStride, width, height, outPixels)
    }

    /**
     * Converts a UINT8 RGB tensor [H,W,3] to ARGB pixels using NEON SIMD.
     * Processes 8 pixels per cycle via vld3/vst4 interleaved loads/stores.
     *
     * @param tensorBuf direct ByteBuffer of the model output (UINT8, RGB order)
     * @param width     output width
     * @param height    output height
     * @param outPixels pre-allocated ARGB output array (size >= width * height)
     * @throws IllegalArgumentException if the buffer is non-direct, dimensions
     *         are invalid, or outPixels is too small.
     */
    private external fun uint8QuantizedTensorToArgbNative(
        tensorBuf: ByteBuffer,
        width: Int,
        height: Int,
        scale: Float,
        zeroPoint: Int,
        outPixels: IntArray
    )

    fun uint8QuantizedTensorToArgb(
        tensorBuf: ByteBuffer,
        width: Int,
        height: Int,
        scale: Float,
        zeroPoint: Int,
        outPixels: IntArray
    ) {
        requireDirect(tensorBuf, "tensorBuf")
        requireDimensions(width, height)
        requireOutputSize(outPixels, width, height)
        uint8QuantizedTensorToArgbNative(
            tensorBuf, width, height, scale, zeroPoint, outPixels
        )
    }

    /**
     * Converts a FLOAT32 RGB tensor [H,W,3] range [-1,1] to ARGB pixels using NEON SIMD.
     * Denormalizes: pixel = (val + 1.0) * 127.5, then clamps to [0,255].
     * Processes 4 pixels per cycle via vld3q/vcvtq float pipeline.
     *
     * @param tensorBuf direct ByteBuffer of the model output (FLOAT32, RGB order)
     * @param width     output width
     * @param height    output height
     * @param outPixels pre-allocated ARGB output array (size >= width * height)
     * @throws IllegalArgumentException if the buffer is non-direct, dimensions
     *         are invalid, or outPixels is too small.
     */
    fun floatTensorToArgb(
        tensorBuf: ByteBuffer,
        width: Int,
        height: Int,
        outPixels: IntArray
    ) {
        requireDirect(tensorBuf, "tensorBuf")
        requireDimensions(width, height)
        requireOutputSize(outPixels, width, height)
        floatTensorToArgbNative(tensorBuf, width, height, outPixels)
    }

    // ── Private native declarations (called only after validation) ────────────

    private external fun yuvToArgbNative(
        yBuf: ByteBuffer,
        uBuf: ByteBuffer,
        vBuf: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        width: Int,
        height: Int,
        outPixels: IntArray
    )

    private external fun uint8TensorToArgbNative(
        tensorBuf: ByteBuffer,
        width: Int,
        height: Int,
        outPixels: IntArray
    )

    private external fun floatTensorToArgbNative(
        tensorBuf: ByteBuffer,
        width: Int,
        height: Int,
        outPixels: IntArray
    )
}