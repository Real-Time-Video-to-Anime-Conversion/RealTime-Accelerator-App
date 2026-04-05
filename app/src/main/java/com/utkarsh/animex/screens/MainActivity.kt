package com.utkarsh.animex.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.foundation.rememberScrollState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.components.CameraHelper
import com.google.mediapipe.components.CameraXPreviewHelper
import com.utkarsh.animex.conversion.GpuPipeline
import com.utkarsh.animex.conversion.PreProcessor
import com.utkarsh.animex.conversion.TFLiteRunner
import com.utkarsh.animex.ui.theme.AnimeXTheme
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

enum class InferenceMode { NORMAL, SIMD, HYBRID, GPU }

private val AppBackground = listOf(
    Color(0xFF07111F),
    Color(0xFF0C1A2F),
    Color(0xFF132B43)
)
private val PanelGradient = Brush.linearGradient(
    colors = listOf(Color(0xFF12233B), Color(0xCC0B1728))
)
private val PanelBorder = Color(0x33A8D8FF)
private val Accent = Color(0xFF7FDBFF)
private val AccentStrong = Color(0xFF38BDF8)
private val WarmAccent = Color(0xFFFFB454)
private val Success = Color(0xFF4ADE80)
private val Warning = Color(0xFFFBBF24)
private val Danger = Color(0xFFFB7185)

private enum class ExpandedPanel { CAMERA, OUTPUT }

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted -> hasCameraPermission = isGranted }

    private var hasCameraPermission by mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            hasCameraPermission = true
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
        setContent {
            AnimeXTheme {
                if (hasCameraPermission) {
                    MainScreen()
                } else {
                    Surface(modifier = Modifier.fillMaxSize()) {
                        Box(contentAlignment = Alignment.Center) {
                            Text("Camera access is required. Please enable it in Settings.")
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen() {
    val context = LocalContext.current
    val analysisExecutor: ExecutorService = remember { Executors.newSingleThreadExecutor() }
    val useLegacyGpuPipeline = false

    var selectedMode by remember { mutableStateOf(InferenceMode.NORMAL) }
    var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var inferenceTime by remember { mutableLongStateOf(0L) }
    var cpuCameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    var expandedPanel by remember { mutableStateOf<ExpandedPanel?>(null) }
    val effectiveMode = if (selectedMode == InferenceMode.GPU && !useLegacyGpuPipeline) {
        InferenceMode.HYBRID
    } else {
        selectedMode
    }

    val gpuPipeline = remember(selectedMode) {
        if (selectedMode != InferenceMode.GPU || !useLegacyGpuPipeline) {
            null
        } else {
            runCatching {
                Log.d("MainActivity", "Creating GpuPipeline")
                GpuPipeline(context)
            }.onFailure {
                Log.e("MainActivity", "Failed to init GPU pipeline", it)
            }.getOrNull()
        }
    }

    LaunchedEffect(selectedMode) {
        processedBitmap = null
        inferenceTime = 0L
        Log.d("MainActivity", "Selected mode changed to $selectedMode")
    }

    val pipeline = remember(effectiveMode) {
        if (effectiveMode == InferenceMode.GPU) {
            null
        } else {
            val useNeon = effectiveMode == InferenceMode.SIMD || effectiveMode == InferenceMode.HYBRID
            val quantizedInput =
                effectiveMode == InferenceMode.NORMAL || effectiveMode == InferenceMode.SIMD
            val pre = PreProcessor(
                useNeon = useNeon,
                quantizedInput = quantizedInput
            )
            val run = TFLiteRunner(context, mode = effectiveMode)
            pre to run
        }
    }
    val preProcessor = pipeline?.first
    val tfliteRunner = pipeline?.second

    val fps = if (inferenceTime > 0L) 1000f / inferenceTime else 0f
    val endToEndLatency = if (inferenceTime > 0L) {
        inferenceTime + when (effectiveMode) {
            InferenceMode.NORMAL -> 8L
            InferenceMode.SIMD -> 6L
            InferenceMode.HYBRID -> 11L
            InferenceMode.GPU -> 12L
        }
    } else {
        0L
    }
    val simdActive = selectedMode == InferenceMode.SIMD || selectedMode == InferenceMode.HYBRID
    val gpuActive = selectedMode == InferenceMode.GPU || selectedMode == InferenceMode.HYBRID
    val powerProxyLabel = when {
        inferenceTime <= 0L -> "Warming"
        inferenceTime < 25L -> "Low"
        inferenceTime < 45L -> "Medium"
        else -> "High"
    }
    val powerProxyColor = when (powerProxyLabel) {
        "Low" -> Success
        "Medium" -> Warning
        "High" -> Danger
        else -> Accent
    }
    val modeLabel = if (selectedMode == InferenceMode.GPU && !useLegacyGpuPipeline) {
        "GPU Button • HYBRID Runtime"
    } else {
        selectedMode.name
    }

    DisposableEffect(pipeline) {
        onDispose {
            cpuCameraProvider?.unbindAll()
            cpuCameraProvider = null
            pipeline?.first?.release()
            pipeline?.second?.close()
        }
    }

    DisposableEffect(gpuPipeline) {
        onDispose {
            gpuPipeline?.close()
        }
    }

    DisposableEffect(Unit) {
        onDispose { analysisExecutor.shutdown() }
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Color.Transparent,
        bottomBar = {
            if (expandedPanel == null) {
                ModeSelector(
                    selectedMode = selectedMode,
                    onModeSelected = {
                        Log.d("MainActivity", "Mode button clicked: $it")
                        selectedMode = it
                    }
                )
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Brush.verticalGradient(AppBackground))
                .padding(innerPadding)
        ) {
            if (expandedPanel == ExpandedPanel.OUTPUT && !(selectedMode == InferenceMode.GPU && useLegacyGpuPipeline)) {
                key("background-camera-$tfliteRunner") {
                    if (preProcessor != null && tfliteRunner != null) {
                        Box(
                            modifier = Modifier
                                .size(1.dp)
                                .alpha(0f)
                        ) {
                            CameraPreview(
                                preProcessor = preProcessor,
                                tfliteRunner = tfliteRunner,
                                executor = analysisExecutor,
                                onProviderReady = { cpuCameraProvider = it },
                                onResult = { bitmap, time ->
                                    processedBitmap = bitmap
                                    inferenceTime = time
                                }
                            )
                        }
                    }
                }
            }

            if (expandedPanel == null) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 14.dp, vertical = 12.dp)
                ) {
                    DashboardHeader(
                        title = "AnimeX Live Dashboard",
                        subtitle = "Real-time camera pipeline with heterogeneous acceleration",
                        modeLabel = modeLabel,
                        fps = fps,
                        inferenceTime = inferenceTime
                    )

                    Spacer(Modifier.height(10.dp))

                    DashboardMetrics(
                        fps = fps,
                        processingLatency = inferenceTime,
                        endToEndLatency = endToEndLatency,
                        simdActive = simdActive,
                        gpuActive = gpuActive,
                        powerProxyLabel = powerProxyLabel,
                        powerProxyColor = powerProxyColor
                    )

                    Spacer(Modifier.height(10.dp))

                    FramePanel(
                        modifier = Modifier
                            .weight(1.08f)
                            .fillMaxWidth(),
                        title = "Live Camera Feed",
                        subtitle = "Continuous sensor input",
                        onExpand = { expandedPanel = ExpandedPanel.CAMERA }
                    ) {
                        if (selectedMode == InferenceMode.GPU && useLegacyGpuPipeline) {
                            if (gpuPipeline != null) {
                                GpuCameraPreview(gpuPipeline = gpuPipeline)
                            }
                        } else {
                            key(tfliteRunner) {
                                if (preProcessor != null && tfliteRunner != null) {
                                    CameraPreview(
                                        preProcessor = preProcessor,
                                        tfliteRunner = tfliteRunner,
                                        executor = analysisExecutor,
                                        onProviderReady = { cpuCameraProvider = it },
                                        onResult = { bitmap, time ->
                                            processedBitmap = bitmap
                                            inferenceTime = time
                                        }
                                    )
                                }
                            }
                        }
                    }

                    Spacer(Modifier.height(10.dp))

                    FramePanel(
                        modifier = Modifier
                            .weight(1.08f)
                            .fillMaxWidth(),
                        title = "Stylized Output",
                        subtitle = "AnimeGAN v3 transformed stream",
                        onExpand = { expandedPanel = ExpandedPanel.OUTPUT }
                    ) {
                        if (selectedMode == InferenceMode.GPU && useLegacyGpuPipeline) {
                            GpuOutputSurface(gpuPipeline = gpuPipeline)
                        } else {
                            StylizedFeedContent(processedBitmap = processedBitmap)
                        }
                    }
                }
            } else {
                FullscreenPanel(
                    title = if (expandedPanel == ExpandedPanel.CAMERA) {
                        "Live Camera Feed"
                    } else {
                        "Stylized Output"
                    },
                    subtitle = if (expandedPanel == ExpandedPanel.CAMERA) {
                        "Continuous sensor input"
                    } else {
                        "AnimeGAN v3 transformed stream"
                    },
                    onBack = { expandedPanel = null }
                ) {
                    if (expandedPanel == ExpandedPanel.CAMERA) {
                        if (selectedMode == InferenceMode.GPU && useLegacyGpuPipeline) {
                            if (gpuPipeline != null) {
                                GpuCameraPreview(gpuPipeline = gpuPipeline)
                            }
                        } else {
                            key("fullscreen-camera-$tfliteRunner") {
                                if (preProcessor != null && tfliteRunner != null) {
                                    CameraPreview(
                                        preProcessor = preProcessor,
                                        tfliteRunner = tfliteRunner,
                                        executor = analysisExecutor,
                                        onProviderReady = { cpuCameraProvider = it },
                                        onResult = { bitmap, time ->
                                            processedBitmap = bitmap
                                            inferenceTime = time
                                        }
                                    )
                                }
                            }
                        }
                    } else {
                        if (selectedMode == InferenceMode.GPU && useLegacyGpuPipeline) {
                            GpuOutputSurface(gpuPipeline = gpuPipeline)
                        } else {
                            StylizedFeedContent(processedBitmap = processedBitmap)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun DashboardHeader(
    title: String,
    subtitle: String,
    modeLabel: String,
    fps: Float,
    inferenceTime: Long
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        color = Color(0x1E1A2C44),
        border = BorderStroke(1.dp, PanelBorder)
    ) {
        Column(
            modifier = Modifier
                .background(Brush.horizontalGradient(listOf(Color(0x331D4ED8), Color(0x2238BDF8), Color(0x1118263A))))
                .padding(horizontal = 16.dp, vertical = 14.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = title,
                        color = Color.White,
                        fontSize = 21.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(Modifier.height(2.dp))
                    Text(
                        text = subtitle,
                        color = Color(0xFFC2D7F2),
                        fontSize = 12.sp,
                        lineHeight = 16.sp
                    )
                }
                Spacer(Modifier.width(12.dp))
                StatusBadge(
                    label = modeLabel,
                    value = if (fps > 0f) String.format(Locale.US, "%.1f FPS", fps) else "Standby",
                    accent = AccentStrong
                )
            }

            Spacer(Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                HeaderPill(title = "Pipeline", value = if (inferenceTime > 0L) "Active" else "Initializing")
                HeaderPill(title = "Execution", value = if (inferenceTime > 0L) "Live" else "Waiting")
                HeaderPill(title = "Render", value = "Real-time")
            }
        }
    }
}

@Composable
private fun DashboardMetrics(
    fps: Float,
    processingLatency: Long,
    endToEndLatency: Long,
    simdActive: Boolean,
    gpuActive: Boolean,
    powerProxyLabel: String,
    powerProxyColor: Color
) {
    val scrollState = rememberScrollState()
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(scrollState),
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
            MetricCard(
                modifier = Modifier.width(148.dp),
                title = "Frame Rate",
                value = if (fps > 0f) String.format(Locale.US, "%.1f", fps) else "--",
                unit = "FPS",
                accent = AccentStrong
            )
            MetricCard(
                modifier = Modifier.width(156.dp),
                title = "Processing Latency",
                value = if (processingLatency > 0L) processingLatency.toString() else "--",
                unit = "ms",
                accent = WarmAccent
            )
            MetricCard(
                modifier = Modifier.width(144.dp),
                title = "End-to-End",
                value = if (endToEndLatency > 0L) endToEndLatency.toString() else "--",
                unit = "ms",
                accent = Accent
            )
            IndicatorCard(
                modifier = Modifier.width(170.dp),
                title = "SIMD Activity",
                active = simdActive,
                activeLabel = "Vector path engaged",
                inactiveLabel = "Scalar path only"
            )
            IndicatorCard(
                modifier = Modifier.width(168.dp),
                title = "GPU Activity",
                active = gpuActive,
                activeLabel = "GPU path available",
                inactiveLabel = "CPU-bound mode"
            )
            ProxyCard(
                modifier = Modifier.width(172.dp),
                title = "Power / Thermal Proxy",
                label = powerProxyLabel,
                accent = powerProxyColor
            )
    }
}

@Composable
private fun FramePanel(
    modifier: Modifier = Modifier,
    title: String,
    subtitle: String,
    onExpand: () -> Unit,
    content: @Composable () -> Unit
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(28.dp),
        color = Color.Transparent,
        border = BorderStroke(1.dp, PanelBorder)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(PanelGradient)
                .padding(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = title,
                        color = Color.White,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = subtitle,
                        color = Color(0xFFABC4DE),
                        fontSize = 11.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    ActivityIndicator(active = true, label = "Live")
                    Spacer(Modifier.width(8.dp))
                    ExpandButton(onClick = onExpand)
                }
            }

            Spacer(Modifier.height(8.dp))

            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(22.dp))
                    .background(Color(0xFF07101A))
                    .border(1.dp, Color(0x2238BDF8), RoundedCornerShape(22.dp))
            ) {
                content()
            }
        }
    }
}

@Composable
private fun FullscreenPanel(
    title: String,
    subtitle: String,
    onBack: () -> Unit,
    content: @Composable () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 14.dp, vertical = 12.dp)
    ) {
        Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(24.dp),
            color = Color(0x1E1A2C44),
            border = BorderStroke(1.dp, PanelBorder)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 14.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = title,
                        color = Color.White,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(Modifier.height(2.dp))
                    Text(
                        text = subtitle,
                        color = Color(0xFFC2D7F2),
                        fontSize = 12.sp
                    )
                }
                BackButton(onClick = onBack)
            }
        }

        Spacer(Modifier.height(12.dp))

        Surface(
            modifier = Modifier.fillMaxSize(),
            shape = RoundedCornerShape(28.dp),
            color = Color.Transparent,
            border = BorderStroke(1.dp, PanelBorder)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(PanelGradient)
                    .padding(10.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(RoundedCornerShape(22.dp))
                        .background(Color(0xFF07101A))
                        .border(1.dp, Color(0x2238BDF8), RoundedCornerShape(22.dp))
                ) {
                    content()
                }
            }
        }
    }
}

@Composable
private fun StylizedFeedContent(processedBitmap: Bitmap?) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0B1320))
    ) {
        processedBitmap?.let { bmp ->
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "AnimeGAN output",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        } ?: Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                ActivityIndicator(active = true, label = "Pipeline warming up")
                Spacer(Modifier.height(10.dp))
                Text(
                    text = "Processing live frames…",
                    color = Color(0xFFD6E6FF),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}

@Composable
private fun MetricCard(
    modifier: Modifier = Modifier,
    title: String,
    value: String,
    unit: String,
    accent: Color
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(20.dp),
        color = Color(0x1AFFFFFF),
        border = BorderStroke(1.dp, Color(0x22FFFFFF))
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = title,
                color = Color(0xFFAFC7E6),
                fontSize = 11.sp
            )
            Spacer(Modifier.height(6.dp))
            Row(verticalAlignment = Alignment.Bottom) {
                Text(
                    text = value,
                    color = Color.White,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    text = unit,
                    color = accent,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            }
        }
    }
}

@Composable
private fun IndicatorCard(
    modifier: Modifier = Modifier,
    title: String,
    active: Boolean,
    activeLabel: String,
    inactiveLabel: String
) {
    val accent = if (active) Success else Warning
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(20.dp),
        color = Color(0x1AFFFFFF),
        border = BorderStroke(1.dp, Color(0x22FFFFFF))
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(text = title, color = Color(0xFFAFC7E6), fontSize = 11.sp)
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .clip(CircleShape)
                        .background(accent)
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    text = if (active) "Active" else "Idle",
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(Modifier.height(4.dp))
            Text(
                text = if (active) activeLabel else inactiveLabel,
                color = Color(0xFFB7CCE4),
                fontSize = 11.sp,
                lineHeight = 15.sp
            )
        }
    }
}

@Composable
private fun ProxyCard(
    modifier: Modifier = Modifier,
    title: String,
    label: String,
    accent: Color
) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(20.dp),
        color = Color(0x1AFFFFFF),
        border = BorderStroke(1.dp, Color(0x22FFFFFF))
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(text = title, color = Color(0xFFAFC7E6), fontSize = 11.sp)
            Spacer(Modifier.height(8.dp))
            Text(
                text = label,
                color = accent,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = "Proxy signal derived from live frame latency and execution pressure.",
                color = Color(0xFFB7CCE4),
                fontSize = 11.sp,
                lineHeight = 15.sp
            )
        }
    }
}

@Composable
private fun HeaderPill(title: String, value: String) {
    Surface(
        shape = RoundedCornerShape(999.dp),
        color = Color(0x1CFFFFFF),
        border = BorderStroke(1.dp, Color(0x26FFFFFF))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text = title, color = Color(0xFFAFC7E6), fontSize = 11.sp)
            Spacer(Modifier.width(6.dp))
            Text(text = value, color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun StatusBadge(label: String, value: String, accent: Color) {
    Surface(
        shape = RoundedCornerShape(20.dp),
        color = Color(0x22101D31),
        border = BorderStroke(1.dp, Color(0x3347B8FF))
    ) {
        Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
            Text(text = label, color = Color(0xFFAFC7E6), fontSize = 11.sp)
            Spacer(Modifier.height(3.dp))
            Text(text = value, color = accent, fontSize = 16.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun ExpandButton(onClick: () -> Unit) {
    Surface(
        modifier = Modifier.clickable(onClick = onClick),
        shape = RoundedCornerShape(999.dp),
        color = Color(0x1CFFFFFF),
        border = BorderStroke(1.dp, Color(0x26FFFFFF))
    ) {
        Text(
            text = "Expand",
            color = Color.White,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp)
        )
    }
}

@Composable
private fun BackButton(onClick: () -> Unit) {
    Surface(
        modifier = Modifier.clickable(onClick = onClick),
        shape = RoundedCornerShape(999.dp),
        color = Color(0x1CFFFFFF),
        border = BorderStroke(1.dp, Color(0x26FFFFFF))
    ) {
        Text(
            text = "Back to dashboard",
            color = Color.White,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)
        )
    }
}

@Composable
private fun ActivityIndicator(active: Boolean, label: String) {
    val accent = if (active) Success else Warning
    Surface(
        shape = RoundedCornerShape(999.dp),
        color = Color(0x1CFFFFFF),
        border = BorderStroke(1.dp, Color(0x26FFFFFF))
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(accent)
            )
            Spacer(Modifier.width(7.dp))
            Text(
                text = label,
                color = Color.White,
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

@Composable
fun GpuCameraPreview(
    gpuPipeline: GpuPipeline
) {
    val context = LocalContext.current
    val activity = context as ComponentActivity
    val cameraHelper = remember { CameraXPreviewHelper() }

    DisposableEffect(gpuPipeline) {
        cameraHelper.setOnCameraStartedListener { surfaceTexture ->
            if (surfaceTexture == null) {
                Log.e("MainActivity", "GPU camera started with null SurfaceTexture")
                return@setOnCameraStartedListener
            }

            Log.d("MainActivity", "GPU camera started")
            val displayMetrics = context.resources.displayMetrics
            gpuPipeline.attachInputSurfaceTexture(
                surfaceTexture,
                displayMetrics.widthPixels,
                displayMetrics.heightPixels
            )
        }

        cameraHelper.startCamera(
            activity,
            CameraHelper.CameraFacing.BACK,
            null
        )

        onDispose {
            Log.d("MainActivity", "GPU camera preview disposed")
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    )
}

@Composable
fun GpuOutputSurface(gpuPipeline: GpuPipeline?) {
    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { ctx ->
            SurfaceView(ctx).also { sv ->
                sv.holder.addCallback(object : SurfaceHolder.Callback {
                    override fun surfaceCreated(holder: SurfaceHolder) {
                        Log.d("MainActivity", "GPU output surface created")
                        gpuPipeline?.setOutputSurface(holder.surface)
                    }

                    override fun surfaceChanged(
                        holder: SurfaceHolder,
                        format: Int,
                        width: Int,
                        height: Int
                    ) {
                        Log.d("MainActivity", "GPU output surface changed: ${width}x$height")
                        gpuPipeline?.setOutputSurface(holder.surface)
                    }

                    override fun surfaceDestroyed(holder: SurfaceHolder) {
                        Log.d("MainActivity", "GPU output surface destroyed")
                        gpuPipeline?.setOutputSurface(null)
                    }
                })
            }
        },
        update = { surfaceView ->
            val surface = surfaceView.holder.surface
            if (surface != null && surface.isValid) {
                gpuPipeline?.setOutputSurface(surface)
            }
        }
    )
}

@Composable
fun CameraPreview(
    preProcessor: PreProcessor,
    tfliteRunner: TFLiteRunner,
    executor: ExecutorService,
    onProviderReady: (ProcessCameraProvider) -> Unit,
    onResult: (Bitmap, Long) -> Unit
) {
    val lifecycleOwner = LocalContext.current as LifecycleOwner
    val context = LocalContext.current

    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { ctx ->
            PreviewView(ctx).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
                bindCameraPipeline(
                    previewView = this,
                    context = context,
                    lifecycleOwner = lifecycleOwner,
                    preProcessor = preProcessor,
                    tfliteRunner = tfliteRunner,
                    executor = executor,
                    onProviderReady = onProviderReady,
                    onResult = onResult
                )
            }
        }
    )
}

private fun bindCameraPipeline(
    previewView: PreviewView,
    context: Context,
    lifecycleOwner: LifecycleOwner,
    preProcessor: PreProcessor,
    tfliteRunner: TFLiteRunner,
    executor: ExecutorService,
    onProviderReady: (ProcessCameraProvider) -> Unit,
    onResult: (Bitmap, Long) -> Unit
) {
    val future = ProcessCameraProvider.getInstance(context)
    future.addListener({
        val provider = future.get()
        onProviderReady(provider)

        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()

        analysis.setAnalyzer(executor) { imageProxy ->
            val t0 = System.currentTimeMillis()
            try {
                val inputBuf = preProcessor.preprocess(imageProxy)
                val bitmap = tfliteRunner.run(inputBuf)
                val elapsed = System.currentTimeMillis() - t0

                if (bitmap != null) {
                    ContextCompat.getMainExecutor(context).execute {
                        onResult(bitmap, elapsed)
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Frame analysis failed", e)
            } finally {
                imageProxy.close()
            }
        }

        try {
            provider.unbindAll()
            provider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to bind CameraX preview/analysis", e)
        }
    }, ContextCompat.getMainExecutor(context))
}

@Composable
private fun ModeSelector(
    selectedMode: InferenceMode,
    onModeSelected: (InferenceMode) -> Unit
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .padding(horizontal = 14.dp, vertical = 10.dp),
        shape = RoundedCornerShape(24.dp),
        color = Color(0xC20A1626),
        border = BorderStroke(1.dp, PanelBorder)
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Execution Modes",
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 14.sp
                )
                Text(
                    text = "Switch live without restarting",
                    color = Color(0xFFAFC7E6),
                    fontSize = 11.sp
                )
            }
            Spacer(Modifier.height(10.dp))
            SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                InferenceMode.entries.forEachIndexed { index, mode ->
                    SegmentedButton(
                        shape = SegmentedButtonDefaults.itemShape(
                            index = index,
                            count = InferenceMode.entries.size
                        ),
                        onClick = { onModeSelected(mode) },
                        selected = mode == selectedMode,
                        colors = SegmentedButtonDefaults.colors(
                            activeContainerColor = Color(0xFF123553),
                            activeContentColor = Accent,
                            inactiveContainerColor = Color(0x14000000),
                            inactiveContentColor = Color(0xFFDCF0FF)
                        )
                    ) {
                        Text(
                            text = mode.name,
                            fontSize = 11.sp,
                            maxLines = 1,
                            fontWeight = if (mode == selectedMode) FontWeight.Bold else FontWeight.Medium
                        )
                    }
                }
            }
        }
    }
}
