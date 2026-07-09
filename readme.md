# Zero-Copy macOS Inference Pipeline

Real-time object detection at **129 FPS** on Apple Silicon using a hardware-level zero-copy memory architecture — AVFoundation → CVPixelBuffer → CoreML/ANE — with no intermediate memory copies between camera capture and neural network inference.

![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)
![Language](https://img.shields.io/badge/language-C%2B%2B17%20%7C%20Objective--C%2B%2B-blue)
![Model](https://img.shields.io/badge/model-YOLOv8n-green)
![FPS](https://img.shields.io/badge/ANE-129%20FPS-brightgreen)

---

## The Problem With Conventional Pipelines

Most inference pipelines on macOS look like this:

```
Webcam → OpenCV VideoCapture → cv::Mat (heap memory)
       → memcpy into ONNX Runtime buffer
       → hidden OS copy into IOSurface before ANE execution
       → inference
```

That's 3+ memory copies of a multi-megabyte frame on every single inference cycle. On Apple Silicon, this is especially wasteful because the CPU and ANE share the same physical DRAM — unified memory — yet the data still gets copied because standard `malloc` produces pageable heap memory that the ANE hardware refuses to read directly.

The Apple Neural Engine requires memory that is pinned (locked in physical RAM), aligned to hardware boundaries, and backed by an `IOSurface`. When you hand it a standard OpenCV `cv::Mat`, macOS intercepts the call and silently copies the entire buffer into an `IOSurface` before inference runs. That hidden copy destroys latency.

---

## The Solution — Hardware-Level Zero-Copy

This pipeline eliminates all intermediate copies by controlling where the frame lives in memory from the moment it leaves the camera sensor:

```
Webcam hardware
    ↓  AVFoundation driver
CVPixelBuffer  (IOSurface-backed — pinned, aligned, hardware-accessible)
    ↓  Vision framework passes pointer directly to
CoreML / Apple Neural Engine  (reads from same physical memory)
    ↓
Detection structs (boxes, classes, confidence)
    ↓  OpenCV draws boxes on converted display frame
Window
```

The pixel data never moves. Same physical memory address from capture to inference.

---

## Architecture

### Why AVFoundation Instead of OpenCV

`cv::VideoCapture` is a cross-platform wrapper — it works on Windows, Linux, Android, and macOS by using the lowest-common-denominator memory model: standard C++ heap allocation via `malloc`. This produces generic, pageable memory with no alignment guarantees.

AVFoundation is Apple's native camera driver. It gives you direct control over which memory block the camera hardware writes into. By configuring `AVCaptureVideoDataOutput` to produce `kCVPixelFormatType_32BGRA` frames, the driver allocates an `IOSurface`-backed `CVPixelBuffer` for each frame automatically.

### Why CVPixelBuffer

A `CVPixelBuffer` is a wrapper around an `IOSurface` — Apple's low-level framework for sharing locked, hardware-aligned memory across the CPU, GPU, and ANE simultaneously. Because M-series chips use unified memory architecture, a buffer allocated this way is directly accessible to all compute units without any cross-bus transfer.

### Why CoreML Instead of ONNX Runtime

ONNX Runtime supports a CoreML Execution Provider, but it goes through an additional abstraction layer. Using CoreML directly via the Vision framework allows passing a `CVPixelBuffer` as the model input natively — the framework is designed for exactly this memory type and routes it to the ANE without staging through CPU memory.

### Objective-C++

AVFoundation and CoreML are Objective-C frameworks. The inference and display logic is C++. The `.mm` extension enables Objective-C++ — Apple's compiler trick that allows both languages in the same file, letting the camera capture and model inference code interoperate with the C++ pipeline and OpenCV drawing layer.

---

## Benchmark Results

Measured over 300 frames on MacBook Pro M4. Each measurement covers the full pipeline: frame capture → inference → NMS → display conversion.

```
═══════════════════════════════════════════════════════
  ANE vs CPU Benchmark  (300 frames each)
═══════════════════════════════════════════════════════

── ANE  (Apple Neural Engine + GPU, via CVPixelBuffer) ──
  Mean:  7.71 ms  →  129 FPS
  p50:   6.65 ms
  p95:   11.41 ms
  p99:   12.88 ms
  Min:   5.01 ms
  Max:   13.85 ms

── CPU  (MLComputeUnitsCPUOnly, standard memory) ──
  Mean:  14.96 ms  →  66 FPS
  p50:   14.77 ms
  p95:   16.41 ms
  p99:   17.90 ms
  Min:   14.12 ms
  Max:   18.67 ms

── Summary ──
  Latency saved:  7.25 ms per frame
  Speedup:        1.94×
  ANE faster by:  48.48%
═══════════════════════════════════════════════════════
```

### Breakdown (Zero-Copy Pipeline)

| Stage                | Time        |
| -------------------- | ----------- |
| CoreML/ANE inference | 9.55 ms     |
| NMS (custom C++)     | < 0.01 ms   |
| Display (cvtColor)   | 0.37 ms     |
| **Total**            | **9.92 ms** |

---

## Technical Stack

| Component       | Technology                                           |
| --------------- | ---------------------------------------------------- |
| Camera capture  | AVFoundation (`AVCaptureSession`)                    |
| Frame memory    | `CVPixelBuffer` (IOSurface-backed)                   |
| Inference       | CoreML + Vision framework                            |
| Compute target  | Apple Neural Engine + GPU (`MLComputeUnitsAll`)      |
| Model           | YOLOv8n (`.mlmodelc`, converted from `.mlpackage`)   |
| Post-processing | Custom C++ NMS                                       |
| Display         | OpenCV (`cv::Mat` wrapping CVPixelBuffer, zero-copy) |
| Build           | CMake 3.20+                                          |
| Language        | C++17 + Objective-C++ (`.mm`)                        |

---

## Project Structure

```
zero-copy-inference/
├── CMakeLists.txt
├── include/
│   ├── CameraCapture.h         # AVFoundation camera interface
│   └── InferencePipeline.h     # CoreML inference interface + Detection struct
├── src/
│   ├── main.mm                 # Main loop — capture, infer, NMS, display
│   ├── CameraCapture.mm        # AVCaptureSession + CVPixelBuffer management
│   ├── InferencePipeline.mm    # VNCoreMLRequest + raw output parsing
│   └── benchmark.mm            # ANE vs CPU latency benchmark
└── models/
    ├── yolov8n.mlmodelc        # Compiled CoreML model (runtime-ready)
    └── yolov8n.mlpackage       # Source CoreML package
```

---

## How It Works — Code-Level

### 1. Camera Capture (CameraCapture.mm)

AVFoundation fires a hardware interrupt (`captureOutput:didOutputSampleBuffer:`) every time the camera captures a frame. The delegate extracts the `CVPixelBuffer` from the sample buffer and retains it with a mutex lock:

```objc
- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    CVPixelBufferRef newBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    std::lock_guard<std::mutex> lock(_bufferMutex);
    if (_currentBuffer) CVPixelBufferRelease(_currentBuffer);
    _currentBuffer = CVPixelBufferRetain(newBuffer);
}
```

The main loop calls `latestPixelBuffer` to get a retained reference to the current frame — the same IOSurface-backed memory the hardware wrote into.

### 2. Inference (InferencePipeline.mm)

The `CVPixelBuffer` is passed directly to a `VNImageRequestHandler`. The Vision framework routes it to CoreML, which schedules execution on the ANE:

```objc
VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
    initWithCVPixelBuffer:pixelBuffer options:@{}];
[handler performRequests:@[request] error:nil];
```

No copy occurs. The ANE reads directly from the IOSurface.

### 3. Output Parsing

YOLOv8n returns a `[1, 84, 8400]` `MLMultiArray`. Rather than using slow NSArray subscript access, a raw C pointer to the underlying data is used for O(1) access:

```cpp
float* data = (float*)output.dataPointer;
int stride1 = output.strides[1].intValue;
int stride2 = output.strides[2].intValue;

// Access class score at row c, box i:
float score = data[c * stride1 + i * stride2];
```

### 4. NMS (main.mm)

Custom greedy NMS filters overlapping detections. Boxes are sorted by confidence, then IoU is computed against all kept boxes:

```cpp
float computeIoU(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, ix2-ix1) * std::max(0.0f, iy2-iy1);
    return inter / ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter);
}
```

### 5. Display (main.mm)

OpenCV wraps the `CVPixelBuffer` base address in a `cv::Mat` header with zero allocation — no copy:

```cpp
CVPixelBufferLockBaseAddress(pixelBuffer, 0);
cv::Mat raw(height, width, CV_8UC4,
            CVPixelBufferGetBaseAddress(pixelBuffer),
            CVPixelBufferGetBytesPerRow(pixelBuffer));
cv::Mat displayFrame;
cv::cvtColor(raw, displayFrame, cv::COLOR_BGRA2BGR);
```

---

## Setup

### Prerequisites

- macOS 13+ on Apple Silicon (M1/M2/M3/M4)
- Xcode 15+ (full install, not just Command Line Tools)
- Homebrew

### Install Dependencies

```bash
brew install opencv cmake
```

### Convert Model

```bash
python3 -m venv venv && source venv/bin/activate
pip install ultralytics coremltools

# Export to CoreML
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='coreml', imgsz=640)"

# Compile for runtime
xcrun coremlcompiler compile yolov8n.mlpackage models/
```

### Build

```bash
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.logicalcpu)
```

### Run

```bash
./inference          # Live detection
./benchmark 300      # ANE vs CPU benchmark over 300 frames
```

---

## Relation to Research Work

This project mirrors the architecture of the NVIDIA Holoscan SDK pipeline used in my SJSU research:

| Research (Holoscan)               | This Project                         |
| --------------------------------- | ------------------------------------ |
| Sony IMX274 → FPGA → RoCE/UDP     | Webcam → AVFoundation                |
| GPUDirect zero-copy to Jetson GPU | CVPixelBuffer zero-copy to Apple ANE |
| CUDA unified memory               | Apple Silicon unified memory         |
| 12 Gbps sustained throughput      | 129 FPS / 7.71ms latency             |

Both systems eliminate the CPU as a data intermediary sensor data flows directly to the compute unit that runs inference.

---

## Author

Sarvesh Koli — MS Computer Engineering, San Jose State University  
[GitHub](https://github.com/Sarves1911) | [LinkedIn](https://www.linkedin.com/in/sarvesh-koli-4805a4231/)
