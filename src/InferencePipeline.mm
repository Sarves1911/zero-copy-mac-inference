#import "InferencePipeline.h"
#import <Vision/Vision.h>

@implementation InferencePipeline {
    VNCoreMLModel *_visionModel;
}

static const std::vector<std::string> CLASS_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

- (instancetype)initWithModelPath:(NSString*)modelPath computeUnits:(MLComputeUnits)units {
    self = [super init];
    if (self) {
        NSURL *modelURL = [NSURL fileURLWithPath:modelPath];
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = units;
        NSError *error = nil;
        MLModel *coreMLModel = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];
        if (coreMLModel) {
            _visionModel = [VNCoreMLModel modelForMLModel:coreMLModel error:&error];
        } else {
            NSLog(@"Failed to load CoreML model: %@", error);
        }
    }
    return self;
}

- (instancetype)initWithModelPath:(NSString*)modelPath {
    return [self initWithModelPath:modelPath computeUnits:MLComputeUnitsAll];
}

- (std::vector<Detection>)runOnPixelBuffer:(CVPixelBufferRef)pixelBuffer origWidth:(int)origWidth origHeight:(int)origHeight {
    __block std::vector<Detection> results;
    if (!_visionModel) return results;

    VNCoreMLRequest *request = [[VNCoreMLRequest alloc] initWithModel:_visionModel completionHandler:^(VNRequest * _Nonnull request, NSError * _Nullable error) {
        NSArray<VNCoreMLFeatureValueObservation*>* observations =
            (NSArray<VNCoreMLFeatureValueObservation*>*)request.results;

        VNCoreMLFeatureValueObservation* obs = observations[0];
        MLMultiArray* output = obs.featureValue.multiArrayValue;
        float* data = (float*)output.dataPointer;
        int stride0 = output.strides[0].intValue;
        int stride1 = output.strides[1].intValue;
        int stride2 = output.strides[2].intValue;

        for (int i = 0; i < 8400; i++) {
            float maxScore = 0.0f;
            int bestClass = -1;
            for (int c = 4; c < 84; c++) {
                float score = data[0*stride0 + c*stride1 + i*stride2];
                if (score > maxScore) { maxScore = score; bestClass = c - 4; }
            }
            if (maxScore < 0.1f) continue;

            float cx = data[0*stride0 + 0*stride1 + i*stride2];
            float cy = data[0*stride0 + 1*stride1 + i*stride2];
            float w  = data[0*stride0 + 2*stride1 + i*stride2];
            float h  = data[0*stride0 + 3*stride1 + i*stride2];

            Detection det;
            det.x1 = cx - w/2;
            det.y1 = cy - h/2;
            det.x2 = cx + w/2;
            det.y2 = cy + h/2;
            det.confidence = maxScore;
            det.className = CLASS_NAMES[bestClass];
            results.push_back(det);
        }
    }];

    request.imageCropAndScaleOption = VNImageCropAndScaleOptionScaleFill;
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
    [handler performRequests:@[request] error:nil];
    return results;
}

@end
