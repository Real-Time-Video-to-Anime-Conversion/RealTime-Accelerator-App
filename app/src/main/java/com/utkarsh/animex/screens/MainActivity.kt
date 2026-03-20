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
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
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
    var inferenceTime by remember { mutableLongStateOf(0L) }
    
    val animeConverter = remember { AnimeConverter(context) }
    val preProcessor = remember { PreProcessor() }

    DisposableEffect(Unit) {
        onDispose {
            animeConverter.close()
        }
    }

    Column(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        // TOP HALF: Original Camera Feed
        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            CameraPreview(
                preProcessor = preProcessor,
                animeConverter = animeConverter,
                onBitmapProcessed = { bitmap, time ->
                    processedBitmap = bitmap
                    inferenceTime = time
                }
            )
            Text(
                "ORIGINAL",
                modifier = Modifier.align(Alignment.TopStart).padding(16.dp).background(Color.Black.copy(alpha = 0.5f)).padding(4.dp),
                color = Color.White,
                fontWeight = FontWeight.Bold
            )
        }

        // BOTTOM HALF: Processed Anime Feed
        Box(modifier = Modifier.weight(1f).fillMaxWidth().background(Color.DarkGray)) {
            processedBitmap?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Anime Filtered View",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop // This ensures it fills the half-screen
                )
            } ?: Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Processing...", color = Color.LightGray)
            }
            
            // Labels and Stats
            Column(modifier = Modifier.align(Alignment.TopStart).padding(16.dp)) {
                Text(
                    "ANIMEGAN V3",
                    modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(4.dp),
                    color = Color.Cyan,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    "Inference: ${inferenceTime}ms",
                    modifier = Modifier.padding(top = 4.dp).background(Color.Black.copy(alpha = 0.5f)).padding(4.dp),
                    color = Color.White,
                    fontSize = 12.sp
                )
            }
        }
    }
}

@Composable
fun CameraPreview(
    preProcessor: PreProcessor,
    animeConverter: AnimeConverter,
    onBitmapProcessed: (Bitmap, Long) -> Unit
) {
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
    onBitmapProcessed: (Bitmap, Long) -> Unit
) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
    val analysisExecutor = Executors.newSingleThreadExecutor()
    val mainExecutor = ContextCompat.getMainExecutor(context)

    cameraProviderFuture.addListener({
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
                val tensorImage = preProcessor.preprocess(imageProxy)
                val resultBitmap = animeConverter.convert(tensorImage)
                val time = animeConverter.lastInferenceTime
                
                mainExecutor.execute {
                    onBitmapProcessed(resultBitmap, time)
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
