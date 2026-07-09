#pragma once
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>

// This class opens the webcam using AVFoundation
// and gives us one CVPixelBuffer per frame
@interface CameraCapture : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>

- (instancetype)init;
- (void)start;
- (void)stop;

// Call this to get the latest frame as a CVPixelBuffer
// Returns NULL if no frame is available yet
- (CVPixelBufferRef)latestPixelBuffer;

@end