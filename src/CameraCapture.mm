#import "CameraCapture.h"
#import <mutex>

@implementation CameraCapture {
    AVCaptureSession *_session;
    CVPixelBufferRef _currentBuffer;
    std::mutex _bufferMutex;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _session = [[AVCaptureSession alloc] init];
        
        // 1. Grab the webcam
        AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:nil];
        if ([_session canAddInput:input]) {
            [_session addInput:input];
        }
        
        // 2. Configure output for 32BGRA (Matches OpenCV/CoreML expectations)
        AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
        output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)};
        
        // 3. Set the delegate to a background hardware thread
        dispatch_queue_t queue = dispatch_queue_create("camera_capture_queue", NULL);
        [output setSampleBufferDelegate:self queue:queue];
        
        if ([_session canAddOutput:output]) {
            [_session addOutput:output];
        }
    }
    return self;
}

- (void)start {
    [_session startRunning];
}

- (void)stop {
    [_session stopRunning];
}

// THE HARDWARE INTERRUPT (Fires every time the webcam captures a new frame)
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    
    CVPixelBufferRef newBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    // Safely lock the memory so we don't overwrite a frame while the AI is reading it
    std::lock_guard<std::mutex> lock(_bufferMutex);
    if (_currentBuffer) {
        CVPixelBufferRelease(_currentBuffer);
    }
    _currentBuffer = CVPixelBufferRetain(newBuffer);
}

// THE C++ CONSUMER API (main.mm calls this)
- (CVPixelBufferRef)latestPixelBuffer {
    std::lock_guard<std::mutex> lock(_bufferMutex);
    if (_currentBuffer) {
        return CVPixelBufferRetain(_currentBuffer); 
    }
    return NULL;
}

@end