#include <jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <arm_neon.h>
#include <algorithm>
#include <tuple>

#define TAG "AnimeX-SIMD"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

extern "C" {

/**
 * Stage 2: YUV_420_888 to RGBA_8888 conversion using ARM NEON
 */
JNIEXPORT void JNICALL
Java_com_utkarsh_animex_conversion_NativeProcessor_yuvToRgbaSIMD(
        JNIEnv *env, jobject thiz,
        jobject y_buffer, jint y_row_stride,
        jobject u_buffer, jint u_row_stride,
        jobject v_buffer, jint v_row_stride,
        jint uv_pixel_stride,
        jobject out_bitmap,
        jint width, jint height) {

    uint8_t *y_ptr = (uint8_t *) env->GetDirectBufferAddress(y_buffer);
    uint8_t *u_ptr = (uint8_t *) env->GetDirectBufferAddress(u_buffer);
    uint8_t *v_ptr = (uint8_t *) env->GetDirectBufferAddress(v_buffer);

    AndroidBitmapInfo info;
    void *pixels;
    AndroidBitmap_getInfo(env, out_bitmap, &info);
    AndroidBitmap_lockPixels(env, out_bitmap, &pixels);

    uint32_t *out_ptr = (uint32_t *) pixels;

    const uint8x16_t v_16 = vdupq_n_u8(16);
    const uint8x16_t v_128 = vdupq_n_u8(128);

    for (int y = 0; y < height; y++) {
        uint8_t *y_row = y_ptr + y * y_row_stride;
        uint8_t *u_row = u_ptr + (y / 2) * u_row_stride;
        uint8_t *v_row = v_ptr + (y / 2) * v_row_stride;
        uint32_t *out_row = out_ptr + y * width;

        for (int x = 0; x < width; x += 16) {
            uint8x16_t vy = vld1q_u8(y_row + x);

            uint8x8_t vu, vv;
            if (uv_pixel_stride == 1) {
                vu = vld1_u8(u_row + x / 2);
                vv = vld1_u8(v_row + x / 2);
            } else {
                uint8x8x2_t u_interleaved = vld2_u8(u_row + x);
                vu = u_interleaved.val[0];
                uint8x8x2_t v_interleaved = vld2_u8(v_row + x);
                vv = v_interleaved.val[0];
            }

            uint8x16_t vu16 = vcombine_u8(vzip1_u8(vu, vu), vzip2_u8(vu, vu));
            uint8x16_t vv16 = vcombine_u8(vzip1_u8(vv, vv), vzip2_u8(vv, vv));

            int16x8_t y_l = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vmaxq_u8(vy, v_16)), vget_low_u8(v_16)));
            int16x8_t y_h = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vmaxq_u8(vy, v_16)), vget_high_u8(v_16)));
            int16x8_t u_l = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vu16), vget_low_u8(v_128)));
            int16x8_t u_h = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vu16), vget_high_u8(v_128)));
            int16x8_t v_l = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vv16), vget_low_u8(v_128)));
            int16x8_t v_h = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(vv16), vget_high_u8(v_128)));

            auto calc_rgb = [](int16x8_t y_s, int16x8_t u_s, int16x8_t v_s) {
                int32x4_t y298_l = vmull_n_s16(vget_low_s16(y_s), 298);
                int32x4_t y298_h = vmull_n_s16(vget_high_s16(y_s), 298);

                int32x4_t r_l = vmlal_n_s16(y298_l, vget_low_s16(v_s), 409);
                int32x4_t r_h = vmlal_n_s16(y298_h, vget_high_s16(v_s), 409);

                int32x4_t g_l = vmlsl_n_s16(y298_l, vget_low_s16(u_s), 100);
                g_l = vmlsl_n_s16(g_l, vget_low_s16(v_s), 208);
                int32x4_t g_h = vmlsl_n_s16(y298_h, vget_high_s16(u_s), 100);
                g_h = vmlsl_n_s16(g_h, vget_high_s16(v_s), 208);

                int32x4_t b_l = vmlal_n_s16(y298_l, vget_low_s16(u_s), 517);
                int32x4_t b_h = vmlal_n_s16(y298_h, vget_high_s16(u_s), 517);

                uint8x8_t r = vqmovun_s16(vcombine_s16(vqrshrn_n_s32(r_l, 8), vqrshrn_n_s32(r_h, 8)));
                uint8x8_t g = vqmovun_s16(vcombine_s16(vqrshrn_n_s32(g_l, 8), vqrshrn_n_s32(g_h, 8)));
                uint8x8_t b = vqmovun_s16(vcombine_s16(vqrshrn_n_s32(b_l, 8), vqrshrn_n_s32(b_h, 8)));

                return std::make_tuple(r, g, b);
            };

            auto res_l = calc_rgb(y_l, u_l, v_l);
            auto res_h = calc_rgb(y_h, u_h, v_h);

            uint8x16x4_t rgba;
            rgba.val[0] = vcombine_u8(std::get<0>(res_l), std::get<0>(res_h));
            rgba.val[1] = vcombine_u8(std::get<1>(res_l), std::get<1>(res_h));
            rgba.val[2] = vcombine_u8(std::get<2>(res_l), std::get<2>(res_h));
            rgba.val[3] = vdupq_n_u8(255);

            vst4q_u8((uint8_t *)(out_row + x), rgba);
        }
    }

    AndroidBitmap_unlockPixels(env, out_bitmap);
}

/**
 * Stage 4: RGB to Grayscale (NEON SIMD)
 */
JNIEXPORT void JNICALL
Java_com_utkarsh_animex_conversion_NativeProcessor_rgbToGrayscaleSIMD(
        JNIEnv *env, jobject thiz,
        jobject bitmap, jobject gray_buffer,
        jint width, jint height) {

    void *pixels;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    uint8_t *src_ptr = (uint8_t *) pixels;
    uint8_t *dst_ptr = (uint8_t *) env->GetDirectBufferAddress(gray_buffer);

    for (int i = 0; i < width * height; i += 16) {
        uint8x16x4_t rgba = vld4q_u8(src_ptr + i * 4);

        uint16x8_t low = vmull_u8(vget_low_u8(rgba.val[0]), vdup_n_u8(77));
        low = vmlal_u8(low, vget_low_u8(rgba.val[1]), vdup_n_u8(150));
        low = vmlal_u8(low, vget_low_u8(rgba.val[2]), vdup_n_u8(29));

        uint16x8_t high = vmull_u8(vget_high_u8(rgba.val[0]), vdup_n_u8(77));
        high = vmlal_u8(high, vget_high_u8(rgba.val[1]), vdup_n_u8(150));
        high = vmlal_u8(high, vget_high_u8(rgba.val[2]), vdup_n_u8(29));

        uint8x16_t gray = vcombine_u8(vshrn_n_u16(low, 8), vshrn_n_u16(high, 8));
        vst1q_u8(dst_ptr + i, gray);
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

/**
 * Stage 5: Sobel Edge Detection (NEON SIMD)
 */
JNIEXPORT void JNICALL
Java_com_utkarsh_animex_conversion_NativeProcessor_sobelEdgeSIMD(
        JNIEnv *env, jobject thiz,
        jobject gray_buffer, jobject edge_buffer,
        jint width, jint height, jint threshold) {

    uint8_t *src = (uint8_t *) env->GetDirectBufferAddress(gray_buffer);
    uint8_t *dst = (uint8_t *) env->GetDirectBufferAddress(edge_buffer);

    for (int y = 1; y < height - 1; y++) {
        uint8_t *r0 = src + (y - 1) * width;
        uint8_t *r1 = src + y * width;
        uint8_t *r2 = src + (y + 1) * width;
        uint8_t *out = dst + y * width;

        for (int x = 1; x < width - 1; x += 8) {
            uint8x8_t p1 = vld1_u8(r0 + x - 1);
            uint8x8_t p2 = vld1_u8(r0 + x);
            uint8x8_t p3 = vld1_u8(r0 + x + 1);
            uint8x8_t p4 = vld1_u8(r1 + x - 1);
            uint8x8_t p6 = vld1_u8(r1 + x + 1);
            uint8x8_t p7 = vld1_u8(r2 + x - 1);
            uint8x8_t p8 = vld1_u8(r2 + x);
            uint8x8_t p9 = vld1_u8(r2 + x + 1);

            int16x8_t gx = vreinterpretq_s16_u16(vsubl_u8(p3, p1));
            gx = vaddq_s16(gx, vshlq_n_s16(vreinterpretq_s16_u16(vsubl_u8(p6, p4)), 1));
            gx = vaddq_s16(gx, vreinterpretq_s16_u16(vsubl_u8(p9, p7)));

            int16x8_t gy = vreinterpretq_s16_u16(vsubl_u8(p7, p1));
            gy = vaddq_s16(gy, vshlq_n_s16(vreinterpretq_s16_u16(vsubl_u8(p8, p2)), 1));
            gy = vaddq_s16(gy, vreinterpretq_s16_u16(vsubl_u8(p9, p3)));

            uint16x8_t edge = vaddq_u16(vreinterpretq_u16_s16(vabsq_s16(gx)),
                                        vreinterpretq_u16_s16(vabsq_s16(gy)));

            uint8x8_t result = vqmovn_u16(vcgtq_u16(edge, vdupq_n_u16(threshold)));
            vst1_u8(out + x, result);
        }
    }
}

/**
 * Stage 7: Compositing ML Output + Edge Map
 */
JNIEXPORT void JNICALL
Java_com_utkarsh_animex_conversion_NativeProcessor_compositeSIMD(
        JNIEnv *env, jobject thiz,
        jobject ml_buffer, jobject edge_buffer, jobject out_bitmap,
        jint width, jint height) {

    uint8_t *ml_ptr = (uint8_t *) env->GetDirectBufferAddress(ml_buffer);
    uint8_t *edge_ptr = (uint8_t *) env->GetDirectBufferAddress(edge_buffer);

    void *pixels;
    AndroidBitmap_lockPixels(env, out_bitmap, &pixels);
    uint8_t *out_ptr = (uint8_t *) pixels;

    for (int i = 0; i < width * height; i += 16) {
        uint8x16x3_t v_ml = vld3q_u8(ml_ptr + i * 3);
        uint8x16_t v_edge = vld1q_u8(edge_ptr + i);

        uint8x16_t v_inv_edge = vmvnq_u8(v_edge);

        auto mult_shift = [&](uint8x16_t channel) {
            uint16x8_t l = vmull_u8(vget_low_u8(channel), vget_low_u8(v_inv_edge));
            uint16x8_t h = vmull_u8(vget_high_u8(channel), vget_high_u8(v_inv_edge));
            return vcombine_u8(vshrn_n_u16(l, 8), vshrn_n_u16(h, 8));
        };

        uint8x16x4_t v_out;
        v_out.val[0] = mult_shift(v_ml.val[0]);
        v_out.val[1] = mult_shift(v_ml.val[1]);
        v_out.val[2] = mult_shift(v_ml.val[2]);
        v_out.val[3] = vdupq_n_u8(255);

        vst4q_u8(out_ptr + i * 4, v_out);
    }

    AndroidBitmap_unlockPixels(env, out_bitmap);
}

}
