#pragma once
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <vector>
#import <string>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int classId;
    std::string className;
};

@interface InferencePipeline : NSObject
- (instancetype)initWithModelPath:(NSString*)modelPath;
- (instancetype)initWithModelPath:(NSString*)modelPath computeUnits:(MLComputeUnits)units;
- (std::vector<Detection>)runOnPixelBuffer:(CVPixelBufferRef)pixelBuffer
                                origWidth:(int)origWidth
                               origHeight:(int)origHeight;
@end
