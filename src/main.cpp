#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    // --- Load model ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "/Users/sarvesh/zero-copy-mac/zero-copy-inference/models/yolov8n.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // --- Grab one frame from webcam ---
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open webcam" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cap >> frame;
    cap.release();

    // --- Preprocess: resize → float → normalize → CHW ---
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 640));
    
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    cv::Mat floatImg;
    rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // Rearrange from HWC to CHW layout
    // HWC: [640, 640, 3] → CHW: [3, 640, 640]
    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);

    // Copy into a flat float buffer in CHW order
    std::vector<float> inputBuffer(1 * 3 * 640 * 640);
    for (int c = 0; c < 3; c++) {
        std::memcpy(inputBuffer.data() + c * 640 * 640,
                    channels[c].data,
                    640 * 640 * sizeof(float));
    }

    // --- Create input tensor pointing at our buffer ---
    std::vector<int64_t> inputShape = {1, 3, 640, 640};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        inputBuffer.data(),
        inputBuffer.size(),
        inputShape.data(),
        inputShape.size()
    );

    // --- Run inference ---
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outputNames, 1
    );

    // --- Print output shape ---
    auto outputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output shape: [";
    for (int i = 0; i < outputShape.size(); i++) {
        std::cout << outputShape[i];
        if (i < outputShape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Inference ran successfully" << std::endl;

    return 0;
}