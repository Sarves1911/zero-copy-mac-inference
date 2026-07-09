cmake_minimum_required(VERSION 3.20)
project(ZeroCopyInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Tell CMake .mm files are Objective-C++
enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(inference
    src/main.mm
    src/CameraCapture.mm
    src/InferencePipeline.mm
)

target_include_directories(inference PRIVATE
    include/
    ${OpenCV_INCLUDE_DIRS}
)

# Link Apple frameworks
target_link_libraries(inference PRIVATE
    ${OpenCV_LIBS}
    "-framework AVFoundation"
    "-framework CoreML"
    "-framework CoreVideo"
    "-framework CoreMedia"
    "-framework Foundation"
    "-framework Vision" 
)

add_executable(benchmark
    src/benchmark.mm
    src/CameraCapture.mm
    src/InferencePipeline.mm
)
target_include_directories(benchmark PRIVATE
    include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(benchmark PRIVATE
    ${OpenCV_LIBS}
    "-framework AVFoundation"
    "-framework CoreML"
    "-framework CoreVideo"
    "-framework CoreMedia"
    "-framework Foundation"
    "-framework Vision"
)
