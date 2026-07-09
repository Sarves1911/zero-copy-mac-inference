
#include <opencv2/opencv.hpp>

#import "CameraCapture.h"
#import "InferencePipeline.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>

#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <thread>

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
using NS        = std::chrono::nanoseconds;

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

static double ns_to_ms(long long ns) { return ns / 1e6; }

struct Stats {
    double mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms, fps;
};

Stats compute_stats(std::vector<long long> samples) {  // pass by value — we sort in place
    std::sort(samples.begin(), samples.end());
    double sum = 0;
    for (auto v : samples) sum += v;
    size_t n = samples.size();
    Stats s;
    s.mean_ms = ns_to_ms(sum / n);
    s.p50_ms  = ns_to_ms(samples[n * 50 / 100]);
    s.p95_ms  = ns_to_ms(samples[n * 95 / 100]);
    s.p99_ms  = ns_to_ms(samples[n * 99 / 100]);
    s.min_ms  = ns_to_ms(samples.front());
    s.max_ms  = ns_to_ms(samples.back());
    s.fps     = 1000.0 / s.mean_ms;
    return s;
}

void print_stats(const char* label, const Stats& s) {
    std::cout << "\n── " << label << " ──\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Mean:  " << s.mean_ms << " ms  →  " << (int)s.fps << " FPS\n";
    std::cout << "  p50:   " << s.p50_ms  << " ms\n";
    std::cout << "  p95:   " << s.p95_ms  << " ms\n";
    std::cout << "  p99:   " << s.p99_ms  << " ms\n";
    std::cout << "  Min:   " << s.min_ms  << " ms\n";
    std::cout << "  Max:   " << s.max_ms  << " ms\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// NMS (identical to main.mm)
// ─────────────────────────────────────────────────────────────────────────────

static float computeIoU(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float aa = (a.x2 - a.x1) * (a.y2 - a.y1);
    float ab = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (aa + ab - inter);
}

static std::vector<Detection> run_nms(std::vector<Detection> dets) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b){ return a.confidence > b.confidence; });
    std::vector<Detection> kept;
    for (const auto& d : dets) {
        bool suppressed = false;
        for (const auto& k : kept)
            if (computeIoU(d, k) > 0.5f) { suppressed = true; break; }
        if (!suppressed) kept.push_back(d);
    }
    return kept;
}

// ─────────────────────────────────────────────────────────────────────────────
// Run one pass of inference + NMS, return nanoseconds
// ─────────────────────────────────────────────────────────────────────────────

static long long run_pass(InferencePipeline* pipeline, CVPixelBufferRef pb) {
    auto t0 = Clock::now();
    auto dets = [pipeline runOnPixelBuffer:pb origWidth:640 origHeight:640];
    run_nms(dets);
    auto t1 = Clock::now();
    return std::chrono::duration_cast<NS>(t1 - t0).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, const char* argv[]) {
    int NUM_FRAMES = 300;
    if (argc > 1) NUM_FRAMES = std::atoi(argv[1]);
    int WARMUP = 30;

    NSString* modelPath = @"/Users/sarvesh/zero-copy-mac/zero-copy-inference/models/yolov8n.mlmodelc";

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  ANE vs CPU Benchmark  (" << NUM_FRAMES << " frames each)\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    // ── Camera ──
    CameraCapture* camera = [[CameraCapture alloc] init];
    [camera start];
    std::this_thread::sleep_for(std::chrono::seconds(1)); // let camera stabilise

    // ── Two pipelines, same model, different compute units ──
    std::cout << "\nLoading ANE pipeline...\n";
    InferencePipeline* ane_pipeline = [[InferencePipeline alloc]
        initWithModelPath:modelPath
             computeUnits:MLComputeUnitsAll];

    std::cout << "Loading CPU pipeline...\n";
    InferencePipeline* cpu_pipeline = [[InferencePipeline alloc]
        initWithModelPath:modelPath
             computeUnits:MLComputeUnitsCPUOnly];

    // ── Storage ──
    std::vector<long long> ane_ns, cpu_ns;
    ane_ns.reserve(NUM_FRAMES);
    cpu_ns.reserve(NUM_FRAMES);

    // ════════════════════════════════════════
    // Pass 1: ANE
    // ════════════════════════════════════════
    std::cout << "\nWarming up ANE (" << WARMUP << " frames)...\n";
    int total = 0, measured = 0;
    while (measured < NUM_FRAMES) {
        CVPixelBufferRef pb = [camera latestPixelBuffer];
        if (!pb) { usleep(1000); continue; }
        long long ns = run_pass(ane_pipeline, pb);
        CVPixelBufferRelease(pb);
        total++;
        if (total <= WARMUP) {
            if (total == WARMUP) std::cout << "Measuring ANE...\n";
            continue;
        }
        ane_ns.push_back(ns);
        measured++;
        if (measured % 100 == 0)
            std::cout << "  ANE " << measured << "/" << NUM_FRAMES << "\n";
    }

    // ════════════════════════════════════════
    // Pass 2: CPU-only
    // ════════════════════════════════════════
    std::cout << "\nWarming up CPU (" << WARMUP << " frames)...\n";
    total = 0; measured = 0;
    while (measured < NUM_FRAMES) {
        CVPixelBufferRef pb = [camera latestPixelBuffer];
        if (!pb) { usleep(1000); continue; }
        long long ns = run_pass(cpu_pipeline, pb);
        CVPixelBufferRelease(pb);
        total++;
        if (total <= WARMUP) {
            if (total == WARMUP) std::cout << "Measuring CPU...\n";
            continue;
        }
        cpu_ns.push_back(ns);
        measured++;
        if (measured % 100 == 0)
            std::cout << "  CPU " << measured << "/" << NUM_FRAMES << "\n";
    }

    [camera stop];

    // ── Results ──
    auto ane_stats = compute_stats(ane_ns);
    auto cpu_stats = compute_stats(cpu_ns);

    double speedup       = cpu_stats.mean_ms / ane_stats.mean_ms;
    double pct_faster    = (1.0 - ane_stats.mean_ms / cpu_stats.mean_ms) * 100.0;
    double ms_saved      = cpu_stats.mean_ms - ane_stats.mean_ms;

    std::cout << "\n\n═══════════════════════════════════════════════════════\n";
    std::cout << "  RESULTS\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    print_stats("ANE  (Apple Neural Engine + GPU)", ane_stats);
    print_stats("CPU  (MLComputeUnitsCPUOnly)",      cpu_stats);

    std::cout << "\n\n═══════════════════════════════════════════════════════\n";
    std::cout << "  SUMMARY\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  ANE latency:     " << ane_stats.mean_ms << " ms  (" << (int)ane_stats.fps << " FPS)\n";
    std::cout << "  CPU latency:     " << cpu_stats.mean_ms << " ms  (" << (int)cpu_stats.fps << " FPS)\n";
    std::cout << "  Latency saved:   " << ms_saved          << " ms per frame\n";
    std::cout << "  Speedup:         " << speedup           << "×\n";
    std::cout << "  ANE is faster:   " << pct_faster        << "%\n";
    std::cout << "    ANE latency:   " << ane_stats.mean_ms << " ms  (" << (int)ane_stats.fps << " FPS)\n";
    std::cout << "    CPU latency:   " << cpu_stats.mean_ms << " ms  (" << (int)cpu_stats.fps << " FPS)\n";
    std::cout << "    Speedup:       " << speedup           << "×\n";
    std::cout << "    % faster:      " << pct_faster        << "%\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";

    return 0;
}