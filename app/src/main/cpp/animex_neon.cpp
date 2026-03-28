#include <jni.h>
#include <arm_neon.h>
#include <android/log.h>
#include <cstring>
#include <cstdint>

#define LOG_TAG "AnimexNeon"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ─────────────────────────────────────────────────────────────────────────────
// YUV 420 888 → ARGB  (NEON, processes 8 pixels per iteration)
// ─────────────────────────────────────────────────────────────────────────────
extern "C"
JNIEXPORT void JNICALL
Java_com_utkarsh_animex_conversion_NeonProcessor_yuvToArgbNative(
        JNIEnv* env,
        jobject /* this */,
        jobject yBuf,
        jobject uBuf,
        jobject vBuf,
        jint    yRowStride,
        jint    uvRowStride,
        jint    uvPixelStride,
        jint    width,
        jint    height,
        jintArray outPixels)
{
    auto* y   = static_cast<const uint8_t*>(env->GetDirectBufferAddress(yBuf));
    auto* u   = static_cast<const uint8_t*>(env->GetDirectBufferAddress(uBuf));
    auto* v   = static_cast<const uint8_t*>(env->GetDirectBufferAddress(vBuf));
    auto* out = static_cast<uint32_t*>(env->GetPrimitiveArrayCritical(outPixels, nullptr));

    if (!y || !u || !v || !out) {
        if (out) env->ReleasePrimitiveArrayCritical(outPixels, out, 0);
        return;
    }

    const int16x8_t v1403 = vdupq_n_s16(1403);
    const int16x8_t v346  = vdupq_n_s16(346);
    const int16x8_t v715  = vdupq_n_s16(715);
    const int16x8_t v1774 = vdupq_n_s16(1774);
    const int16x8_t v128  = vdupq_n_s16(128);
    const uint8x8_t v255  = vdup_n_u8(255);

    for (int row = 0; row < height; ++row) {
        const uint8_t* yRow  = y + yRowStride  * row;
        const uint8_t* uvRow = u + uvRowStride * (row >> 1);
        const uint8_t* vRow  = v + uvRowStride * (row >> 1);

        int col = 0;
        for (; col <= width - 8; col += 8) {
            uint8x8_t yVals = vld1_u8(yRow + col);
            uint8x8_t uVals4, vVals4;

            // OPTIMIZED UV LOADING: No more scalar loops or tmp arrays.
            if (uvPixelStride == 1) {
                // Planar: Load 8 bytes (enough for 16 pixels), but we use 4 for 8 pixels.
                uint8x8_t u4 = vld1_u8(uvRow + (col >> 1));
                uint8x8_t v4 = vld1_u8(vRow + (col >> 1));
                // Duplicate each UV for 2 horizontal Y pixels: [u0,u0,u1,u1...]
                uVals4 = vzip_u8(u4, u4).val[0];
                vVals4 = vzip_u8(v4, v4).val[0];
            } else {
                // Interleaved (NV21/NV12): Load 8 bytes interleaved, de-interleave to 4 U and 4 V
                uint8x8x2_t uv8 = vld2_u8(uvRow + (col >> 1) * uvPixelStride);
                uVals4 = vzip_u8(uv8.val[0], uv8.val[0]).val[0];
                vVals4 = vzip_u8(uv8.val[1], uv8.val[1]).val[0];
            }

            int16x8_t yS = vreinterpretq_s16_u16(vmovl_u8(yVals));
            int16x8_t uS = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(uVals4)), v128);
            int16x8_t vS = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vVals4)), v128);

            int16x8_t rS = vaddq_s16(yS, vshrq_n_s16(vmulq_s16(v1403, vS), 10));
            int16x8_t gS = vsubq_s16(vsubq_s16(yS, vshrq_n_s16(vmulq_s16(v346, uS), 10)), vshrq_n_s16(vmulq_s16(v715, vS), 10));
            int16x8_t bS = vaddq_s16(yS, vshrq_n_s16(vmulq_s16(v1774, uS), 10));

            uint8x8_t rU = vqmovun_s16(rS);
            uint8x8_t gU = vqmovun_s16(gS);
            uint8x8_t bU = vqmovun_s16(bS);

            uint8x8x4_t argb;
            argb.val[0] = bU;
            argb.val[1] = gU;
            argb.val[2] = rU;
            argb.val[3] = v255;

            vst4_u8(reinterpret_cast<uint8_t*>(out + row * width + col), argb);
        }
        // Scalar tail for remaining pixels
        for (; col < width; ++col) {
            int uvIdx = (col >> 1) * uvPixelStride;
            int yVal  = yRow[col];
            int r = yVal + (1403 * (vRow[uvIdx] - 128) >> 10);
            int g = yVal - (346 * (uvRow[uvIdx] - 128) >> 10) - (715 * (vRow[uvIdx] - 128) >> 10);
            int b = yVal + (1774 * (uvRow[uvIdx] - 128) >> 10);
            r = r < 0 ? 0 : r > 255 ? 255 : r;
            g = g < 0 ? 0 : g > 255 ? 255 : g;
            b = b < 0 ? 0 : b > 255 ? 255 : b;
            out[row * width + col] = (0xFF << 24) | (r << 16) | (g << 8) | b;
        }
    }
    env->ReleasePrimitiveArrayCritical(outPixels, out, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// UINT8 tensor [H, W, 3] → ARGB int array  (NEON, 8 pixels per iteration)
// ─────────────────────────────────────────────────────────────────────────────
extern "C"
JNIEXPORT void JNICALL
Java_com_utkarsh_animex_conversion_NeonProcessor_uint8TensorToArgbNative(
        JNIEnv* env,
        jobject /* this */,
        jobject tensorBuf,
        jint    width,
        jint    height,
        jintArray outPixels)
{
    auto* src = static_cast<const uint8_t*>(env->GetDirectBufferAddress(tensorBuf));
    auto* out = static_cast<uint32_t*>(env->GetPrimitiveArrayCritical(outPixels, nullptr));

    if (!src || !out || width <= 0 || height <= 0) {
        if (out) env->ReleasePrimitiveArrayCritical(outPixels, out, 0);
        return;
    }

    const uint8x8_t alpha = vdup_n_u8(255);
    int totalPixels = width * height;
    int i = 0;

    for (; i <= totalPixels - 8; i += 8) {
        uint8x8x3_t rgb = vld3_u8(src + i * 3);
        uint8x8x4_t argb;
        argb.val[0] = rgb.val[2];
        argb.val[1] = rgb.val[1];
        argb.val[2] = rgb.val[0];
        argb.val[3] = alpha;
        vst4_u8(reinterpret_cast<uint8_t*>(out + i), argb);
    }
    for (; i < totalPixels; ++i) {
        out[i] = (0xFF << 24) | (src[i*3] << 16) | (src[i*3+1] << 8) | src[i*3+2];
    }
    env->ReleasePrimitiveArrayCritical(outPixels, out, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// FLOAT32 tensor [H, W, 3] range [-1,1] → ARGB int array (NEON, 4 pixels/iter)
// ─────────────────────────────────────────────────────────────────────────────
extern "C"
JNIEXPORT void JNICALL
Java_com_utkarsh_animex_conversion_NeonProcessor_floatTensorToArgbNative(
        JNIEnv* env,
        jobject /* this */,
        jobject tensorBuf,
        jint    width,
        jint    height,
        jintArray outPixels)
{
    auto* src = static_cast<const float*>(env->GetDirectBufferAddress(tensorBuf));
    auto* out = static_cast<uint32_t*>(env->GetPrimitiveArrayCritical(outPixels, nullptr));

    if (!src || !out || width <= 0 || height <= 0) {
        if (out) env->ReleasePrimitiveArrayCritical(outPixels, out, 0);
        return;
    }

    const float32x4_t vOne   = vdupq_n_f32(1.0f);
    const float32x4_t v127_5 = vdupq_n_f32(127.5f);

    int totalPixels = width * height;
    int i = 0;

    for (; i <= totalPixels - 4; i += 4) {
        float32x4x3_t rgb = vld3q_f32(src + i * 3);
        float32x4_t rF = vmulq_f32(vaddq_f32(rgb.val[0], vOne), v127_5);
        float32x4_t gF = vmulq_f32(vaddq_f32(rgb.val[1], vOne), v127_5);
        float32x4_t bF = vmulq_f32(vaddq_f32(rgb.val[2], vOne), v127_5);

        uint8x8_t rU = vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(rF)), vdup_n_u16(0)));
        uint8x8_t gU = vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(gF)), vdup_n_u16(0)));
        uint8x8_t bU = vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(bF)), vdup_n_u16(0)));

        uint8x8x4_t argb;
        argb.val[0] = bU;
        argb.val[1] = gU;
        argb.val[2] = rU;
        argb.val[3] = vdup_n_u8(255);
        uint8_t* dst = reinterpret_cast<uint8_t*>(out + i);
        vst4_lane_u8(dst +  0, argb, 0);
        vst4_lane_u8(dst +  4, argb, 1);
        vst4_lane_u8(dst +  8, argb, 2);
        vst4_lane_u8(dst + 12, argb, 3);
    }
    for (; i < totalPixels; ++i) {
        auto clamp = [](float v) -> uint8_t {
            int iv = static_cast<int>((v + 1.0f) * 127.5f);
            return static_cast<uint8_t>(iv < 0 ? 0 : iv > 255 ? 255 : iv);
        };
        out[i] = (0xFF << 24) | (clamp(src[i*3]) << 16) | (clamp(src[i*3+1]) << 8) | clamp(src[i*3+2]);
    }

    env->ReleasePrimitiveArrayCritical(outPixels, out, 0);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_utkarsh_animex_conversion_NeonProcessor_uint8QuantizedTensorToArgbNative(
        JNIEnv* env,
        jobject /* this */,
        jobject tensorBuf,
        jint width,
        jint height,
        jfloat scale,
        jint zeroPoint,
        jintArray outPixels)
{
    auto* src = static_cast<const uint8_t*>(env->GetDirectBufferAddress(tensorBuf));
    auto* out = static_cast<uint32_t*>(env->GetPrimitiveArrayCritical(outPixels, nullptr));

    if (!src || !out || width <= 0 || height <= 0) {
        if (out) env->ReleasePrimitiveArrayCritical(outPixels, out, 0);
        return;
    }

    const int totalPixels = width * height;

    // pixel = ((q - zeroPoint) * scale + 1.0) * 127.5
    const float alpha = scale * 127.5f;

    const int16x8_t vZeroPoint = vdupq_n_s16(static_cast<int16_t>(zeroPoint));
    const float32x4_t vAlpha = vdupq_n_f32(alpha);
    const float32x4_t vBias = vdupq_n_f32(127.5f);
    const float32x4_t vZeroF = vdupq_n_f32(0.0f);
    const float32x4_t v255F = vdupq_n_f32(255.0f);
    const uint8x8_t v255U8 = vdup_n_u8(255);

    auto dequantize8 = [&](uint8x8_t qVals) -> uint8x8_t {
        int16x8_t centered =
                vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(qVals)), vZeroPoint);

        int32x4_t lo32 = vmovl_s16(vget_low_s16(centered));
        int32x4_t hi32 = vmovl_s16(vget_high_s16(centered));

        float32x4_t loF = vaddq_f32(vmulq_f32(vcvtq_f32_s32(lo32), vAlpha), vBias);
        float32x4_t hiF = vaddq_f32(vmulq_f32(vcvtq_f32_s32(hi32), vAlpha), vBias);

        loF = vmaxq_f32(vZeroF, vminq_f32(loF, v255F));
        hiF = vmaxq_f32(vZeroF, vminq_f32(hiF, v255F));

        int32x4_t loI = vcvtq_s32_f32(loF);
        int32x4_t hiI = vcvtq_s32_f32(hiF);

        uint16x4_t lo16 = vqmovun_s32(loI);
        uint16x4_t hi16 = vqmovun_s32(hiI);

        return vmovn_u16(vcombine_u16(lo16, hi16));
    };

    int i = 0;
    for (; i <= totalPixels - 8; i += 8) {
        uint8x8x3_t rgb = vld3_u8(src + i * 3);

        uint8x8_t rU8 = dequantize8(rgb.val[0]);
        uint8x8_t gU8 = dequantize8(rgb.val[1]);
        uint8x8_t bU8 = dequantize8(rgb.val[2]);

        uint8x8x4_t bgra;
        bgra.val[0] = bU8;
        bgra.val[1] = gU8;
        bgra.val[2] = rU8;
        bgra.val[3] = v255U8;

        vst4_u8(reinterpret_cast<uint8_t*>(out + i), bgra);
    }

    auto dequantScalar = [scale, zeroPoint](uint8_t q) -> uint8_t {
        float real = (static_cast<int>(q) - zeroPoint) * scale;
        int pixel = static_cast<int>((real + 1.0f) * 127.5f);
        return static_cast<uint8_t>(pixel < 0 ? 0 : pixel > 255 ? 255 : pixel);
    };

    for (; i < totalPixels; ++i) {
        uint8_t r = dequantScalar(src[i * 3]);
        uint8_t g = dequantScalar(src[i * 3 + 1]);
        uint8_t b = dequantScalar(src[i * 3 + 2]);

        out[i] = (0xFFu << 24) |
                 (static_cast<uint32_t>(r) << 16) |
                 (static_cast<uint32_t>(g) << 8) |
                 static_cast<uint32_t>(b);
    }

    env->ReleasePrimitiveArrayCritical(outPixels, out, 0);
}


