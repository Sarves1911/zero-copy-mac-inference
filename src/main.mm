#include <opencv2/opencv.hpp>
#include "CameraCapture.h"
#include "InferencePipeline.h"
#include <iostream>
#include <chrono>



float computeIoU(const Detection& a, const Detection& b)
{
    float intersect_x1 = std::max(a.x1, b.x1);
    float intersect_y1 = std::max(a.y1, b.y1);
    float intersect_x2 = std::min(a.x2, b.x2);
    float intersect_y2 = std::min(a.y2, b.y2);

    float intersection = std::max(0.0f, intersect_x2 - intersect_x1) * std::max(0.0f, intersect_y2 - intersect_y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float iou = intersection / (area_a + area_b - intersection);

    return iou;

}

int main() {
    CameraCapture* camera = [[CameraCapture alloc] init];
    [camera start];

    NSString* modelPath = @"/Users/sarvesh/zero-copy-mac/zero-copy-inference/models/yolov8n.mlmodelc";
    InferencePipeline* ai = [[InferencePipeline alloc] initWithModelPath:modelPath];

    cv::namedWindow("Zero-Copy Inference", cv::WINDOW_AUTOSIZE);
    std::cout << "Starting native hardware loop. Press 'q' to quit." << std::endl;

    while (true) {
        CVPixelBufferRef pixelBuffer = [camera latestPixelBuffer];
        if (!pixelBuffer) continue;
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<Detection> detections = [ai runOnPixelBuffer:pixelBuffer origWidth:640 origHeight:640];

        CVPixelBufferLockBaseAddress(pixelBuffer, 0);

        void* baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
        size_t width = CVPixelBufferGetWidth(pixelBuffer);
        size_t height = CVPixelBufferGetHeight(pixelBuffer);
        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);

        cv::Mat raw(height, width, CV_8UC4, baseAddress, bytesPerRow);

        // Convert BGRA to BGR so OpenCV can display it correctly
        cv::Mat displayFrame;
        cv::cvtColor(raw, displayFrame, cv::COLOR_BGRA2BGR);

        // Scale boxes from 640x640 to actual frame size and draw
        float scaleX = (float)width / 640.0f;
        float scaleY = (float)height / 640.0f;
        // Sort and NMS
        std::sort(detections.begin(), detections.end(),
                [](const Detection& a, const Detection& b) {
                    return a.confidence > b.confidence;
                });

        std::vector<Detection> kept;
        for (const auto& det : detections) {
            bool suppressed = false;
            for (const auto& k : kept) {
                if (computeIoU(det, k) > 0.5f) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) kept.push_back(det);
        }
        //std::cout << "Before NMS: " << detections.size() << " After NMS: " << kept.size() << std::endl;
        //if (!kept.empty()) {
        // std::cout << "First kept box: x1=" << kept[0].x1 << " y1=" << kept[0].y1 
        //       << " x2=" << kept[0].x2 << " y2=" << kept[0].y2 << std::endl;
        //     }

        for (const auto& det : kept) {
            int x1 = (int)(det.x1 * scaleX);
            int y1 = (int)(det.y1 * scaleY);
            int x2 = (int)(det.x2 * scaleX);
            int y2 = (int)(det.y2 * scaleY);
            cv::rectangle(displayFrame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            cv::putText(displayFrame, det.className, cv::Point(x1, y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }

        // std::cout << "Drawing " << detections.size() << " boxes, frame size: " 
        //   << displayFrame.cols << "x" << displayFrame.rows << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        double fps = 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        //std::string fpsText = "FPS: " + std::to_string((int)fps);
        std::cout << fpsText << std::endl;
        cv::putText(displayFrame, fpsText, cv::Point(20, 60),
            cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 255, 255), 3);
        cv::imshow("Zero-Copy Inference", displayFrame);

        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
        CVPixelBufferRelease(pixelBuffer);

        if (cv::waitKey(1) == 'q') break;
    }

    [camera stop];
    return 0;
}
