#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

const int kWarmUpTurns = 10;
const size_t kStepFactor = 2;  // Data size increases by powers of 2 in each step
const size_t kMinSize = 1 * 1024;   // Minimum size: 1 KB
const size_t kMaxSize = 512 * 1024 * 1024; // Maximum size: 512 MB

void WarmUpDevice(int device_id) {
    cudaSetDevice(device_id);

    int* pointer;
    size_t warmup_size = kMinSize;
    cudaMalloc(&pointer, warmup_size);
    cudaMemset(pointer, 0, warmup_size);
    cudaFree(pointer);
}

float MeasureBandwidthD2D(int device_id0, int device_id1, size_t size, bool peer_access = false) {
    int *pointer0, *pointer1;

    // Enable Peer Access if requested
    if (peer_access) {
        cudaSetDevice(device_id0);
        cudaDeviceEnablePeerAccess(device_id1, 0);
    }

    cudaSetDevice(device_id0);
    cudaMalloc(&pointer0, size);

    cudaSetDevice(device_id1);
    cudaMalloc(&pointer1, size);

    cudaSetDevice(device_id0);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // Warm-up (to mitigate driver overhead for initial transfers)
    for (int i = 0; i < kWarmUpTurns; i++) {
        cudaMemcpy(pointer0, pointer1, size, cudaMemcpyDeviceToDevice);
    }

    // Start measurement
    cudaEventRecord(begin);
    cudaMemcpy(pointer0, pointer1, size, cudaMemcpyDeviceToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, begin, end);
    elapsed /= 1000.0f;  // Convert to seconds

    cudaFree(pointer0);
    cudaFree(pointer1);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    return (size / (1024.0 * 1024.0 * 1024.0)) / elapsed;  // Bandwidth in GB/s
}

float MeasureBandwidthH2D(int device_id, size_t size) {
    int* dev_ptr;
    int* host_ptr;

    cudaSetDevice(device_id);
    cudaHostAlloc((void**)&host_ptr, size, cudaHostAllocDefault);
    cudaMalloc(&dev_ptr, size);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // Warm-up
    for (int i = 0; i < kWarmUpTurns; i++) {
        cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    }

    // Start measurement
    cudaEventRecord(begin);
    cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, begin, end);
    elapsed /= 1000.0f;  // Convert to seconds

    cudaFreeHost(host_ptr);
    cudaFree(dev_ptr);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    return (size / (1024.0 * 1024.0 * 1024.0)) / elapsed;  // Bandwidth in GB/s
}

int main() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    if (numGPUs < 2) {
        std::cout << "Error: At least two GPUs are required." << std::endl;
        return 1;
    }

    for (int device = 0; device < numGPUs; ++device) {
        WarmUpDevice(device);
    }

    std::cout << "Bandwidth test (P2P and D2D):" << std::endl;
    std::cout << "Size (KB)\tGPU1->GPU2 (D2D)\tGPU1->GPU2 (P2P)" << std::endl;

    for (size_t size = kMinSize; size <= kMaxSize; size *= kStepFactor) {
        float d2d_bandwidth = MeasureBandwidthD2D(0, 1, size);
        float p2p_bandwidth = MeasureBandwidthD2D(0, 1, size, true);

        std::cout << size / 1024 << "\t" << d2d_bandwidth << "\t" << p2p_bandwidth << std::endl;
    }

    std::cout << "\nPinned Memory H2D Bandwidth:" << std::endl;
    std::cout << "Size (KB)\tH2D Bandwidth (GB/s)" << std::endl;

    for (size_t size = kMinSize; size <= kMaxSize; size *= kStepFactor) {
        float h2d_bandwidth = MeasureBandwidthH2D(0, size);
        std::cout << size / 1024 << "\t" << h2d_bandwidth << std::endl;
    }

    return 0;
}

