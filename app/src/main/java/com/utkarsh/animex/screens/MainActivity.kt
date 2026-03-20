package com.utkarsh.animex.screens

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import com.utkarsh.animex.ui.theme.AnimeXTheme
import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.activity.result.contract.ActivityResultContracts
import android.Manifest
import android.content.pm.PackageManager
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import com.utkarsh.animex.conversion.AnimeConverter
import com.utkarsh.animex.conversion.PreProcessor
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    private var hasCameraPermission by mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            hasCameraPermission = true
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
        setContent {
            AnimeXTheme {
                if (hasCameraPermission) {
                    MainScreen()
                } else {
                    Text("Camera access is required to use this app. Please enable it in settings")
                }
            }
        }
    }
}

@Composable
fun MainScreen() {
    val context = LocalContext.current
    var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    
    // Manage converter lifecycle
    val animeConverter = remember { AnimeConverter(context) }
    val preProcessor = remember { PreProcessor() }

    DisposableEffect(Unit) {
        onDispose {
            animeConverter.close()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // CameraPreview runs in the background to feed the analysis pipeline
        CameraPreview(
            preProcessor = preProcessor,
            animeConverter = animeConverter,
            onBitmapProcessed = { bitmap ->
                processedBitmap = bitmap
            }
        )

        // Show the final processed bitmap on top of the preview
        processedBitmap?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Anime Filtered View",
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}

@Composable
fun CameraPreview(
    preProcessor: PreProcessor,
    animeConverter: AnimeConverter,
    onBitmapProcessed: (Bitmap) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalContext.current as LifecycleOwner

    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { ctx ->
            PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                startCameraPipeline(this, ctx, lifecycleOwner, preProcessor, animeConverter, onBitmapProcessed)
            }
        }
    )
}

fun startCameraPipeline(
    previewView: PreviewView,
    context: Context,
    lifecycleOwner: LifecycleOwner,
    preProcessor: PreProcessor,
    animeConverter: AnimeConverter,
    onBitmapProcessed: (Bitmap) -> Unit
) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    val analysisExecutor = Executors.newSingleThreadExecutor()
    val mainExecutor = ContextCompat.getMainExecutor(context)

    cameraProviderFuture.addListener({
        val cameraProvider = cameraProviderFuture.get()

        // 1. Preview Use Case
        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        // 2. Image Analysis Use Case (The Pipeline)
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()

        imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy ->
            try {
                // STEP 1: Preprocess (YUV -> Bitmap -> TensorImage)
                val tensorImage = preProcessor.preprocess(imageProxy)
                
                // STEP 2: Convert (Model Inference)
                val resultBitmap = animeConverter.convert(tensorImage)
                
                // STEP 3: Update UI on Main Thread
                mainExecutor.execute {
                    onBitmapProcessed(resultBitmap)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                imageProxy.close()
            }
        }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysis
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }

    }, mainExecutor)
}
