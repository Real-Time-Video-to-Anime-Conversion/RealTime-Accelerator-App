package com.utkarsh.animex.conversion

import android.content.Context
import android.graphics.SurfaceTexture
import android.util.Log
import android.view.Surface
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.glutil.EglManager

private const val TAG = "GpuPipeline"
private const val GRAPH_FILE = "animegan_gpu.binarypb"
private const val INPUT_STREAM = "input_video"
private const val OUTPUT_STREAM = "output_video"

class GpuPipeline(private val context: Context) {

    companion object {
        init {
            System.loadLibrary("mediapipe_jni")
        }
    }

    private val eglManager: EglManager
    private val processor: FrameProcessor
    private val converter: ExternalTextureConverter
    @Volatile
    private var isClosed = false

    init {
        AndroidAssetUtil.initializeNativeAssetManager(context)

        eglManager = EglManager(null)

        processor = FrameProcessor(
            context,
            eglManager.nativeContext,
            GRAPH_FILE,
            INPUT_STREAM,
            OUTPUT_STREAM
        )

        val videoOutput = requireNotNull(processor.videoSurfaceOutput) {
            "MediaPipe graph did not create a videoSurfaceOutput"
        }
        videoOutput.setFlipY(true)

        converter = ExternalTextureConverter(eglManager.context, 2)
        converter.setFlipY(true)
        converter.setConsumer(processor)

        Log.d(TAG, "GpuPipeline initialized")
    }

    fun attachInputSurfaceTexture(
        surfaceTexture: SurfaceTexture,
        width: Int,
        height: Int
    ) {
        if (isClosed) return
        Log.d(TAG, "attachInputSurfaceTexture ${width}x$height")
        converter.setSurfaceTextureAndAttachToGLContext(surfaceTexture, width, height)
    }

    fun setOutputSurface(surface: Surface?) {
        if (isClosed) return
        Log.d(TAG, "setOutputSurface(surface=${surface != null})")
        processor.videoSurfaceOutput?.setSurface(surface)
    }

    fun close() {
        if (isClosed) return
        isClosed = true

        Log.d(TAG, "Closing GpuPipeline")
        processor.videoSurfaceOutput?.setSurface(null)
        converter.close()
        processor.close()
        eglManager.release()
    }
}
