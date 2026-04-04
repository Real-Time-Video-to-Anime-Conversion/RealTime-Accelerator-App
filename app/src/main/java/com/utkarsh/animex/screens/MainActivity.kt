package com.utkarsh.animex.screens

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.*
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
import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.LocalLifecycleOwner
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
                    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text("Camera access is required. Please enable it in settings")
                    }
                }
            }
        }
    }
}

@Composable
fun MainScreen() {
    val context = LocalContext.current
    var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var useSIMD by remember { mutableStateOf(false) }
    
    val animeConverter = remember { 
        try {
            AnimeConverter(context)
        } catch (e: Exception) {
            Log.e("AnimeX", "Failed to initialize converter", e)
            null
        }
    }
    val preProcessor = remember { PreProcessor() }

    DisposableEffect(animeConverter) {
        onDispose {
            animeConverter?.close()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        if (animeConverter != null) {
            CameraPreview(
                preProcessor = preProcessor,
                animeConverter = animeConverter,
                useSIMD = useSIMD,
                onBitmapProcessed = { bitmap ->
                    processedBitmap = bitmap
                }
            )

            processedBitmap?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Anime Filtered View",
                    modifier = Modifier.fillMaxSize()
                )
            }

            // Mode Toggle UI
            Column(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.8f)
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = if (useSIMD) "MODE: SIMD (NEON)" else "MODE: BASELINE",
                            style = MaterialTheme.typography.labelLarge,
                            color = if (useSIMD) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary
                        )
                        Spacer(modifier = Modifier.width(16.dp))
                        Switch(
                            checked = useSIMD,
                            onCheckedChange = { useSIMD = it }
                        )
                    }
                }
                Text(
                    text = if (useSIMD) "Processing with ARM NEON" else "Processing with Kotlin Loops",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.padding(top = 8.dp)
                )
            }
        } else {
            Text("Error loading AI model.")
        }
    }
}

@Composable
fun CameraPreview(
    preProcessor: PreProcessor,
    animeConverter: AnimeConverter,
    useSIMD: Boolean,
    onBitmapProcessed: (Bitmap) -> Unit
) {
    val lifecycleOwner = LocalLifecycleOwner.current

    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { ctx ->
            PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                startCameraPipeline(this, ctx, lifecycleOwner, preProcessor, animeConverter, useSIMD, onBitmapProcessed)
            }
        },
        update = { previewView ->
            // Re-bind when mode changes
            startCameraPipeline(previewView, previewView.context, lifecycleOwner, preProcessor, animeConverter, useSIMD, onBitmapProcessed)
        }
    )
}

fun startCameraPipeline(
    previewView: PreviewView,
    context: Context,
    lifecycleOwner: LifecycleOwner,
    preProcessor: PreProcessor,
    animeConverter: AnimeConverter,
    useSIMD: Boolean,
    onBitmapProcessed: (Bitmap) -> Unit
) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    val analysisExecutor = Executors.newSingleThreadExecutor()
    val mainExecutor = ContextCompat.getMainExecutor(context)

    cameraProviderFuture.addListener({
        try {
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy ->
                try {
                    // Pass the useSIMD flag down the pipeline
                    val (tensorImage, originalBitmap) = preProcessor.preprocess(imageProxy, useSIMD)
                    val resultBitmap = animeConverter.convert(tensorImage, originalBitmap, useSIMD)
                    
                    mainExecutor.execute {
                        onBitmapProcessed(resultBitmap)
                    }
                } catch (e: Exception) {
                    Log.e("AnimeX", "Pipeline error", e)
                } finally {
                    imageProxy.close()
                }
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
        } catch (e: Exception) {
            Log.e("AnimeX", "Camera binding failed", e)
        }
    }, mainExecutor)
}
