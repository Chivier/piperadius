#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

const int kWarmUpTurns = 10;
const size_t kWarmUpSize = ((size_t) 2) * 1024 * 1024 * 1024; // 2GB

float DirectCopyTest(int device_id0, int device_id1, size_t size) { int *pointers[2];

    cudaSetDevice(device_id0);
    // cudaDeviceEnablePeerAccess(device_id1, 0);
    cudaMalloc(&pointers[0], size);

    cudaSetDevice(device_id1);
    // cudaDeviceEnablePeerAccess(device_id0, 0);
    cudaMalloc(&pointers[1], size);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // Check available
    // int avail;
    // cudaDeviceCanAccessPeer(&avail, device_id0, device_id1);
    // printf("[%d]", avail);

    // Warm Up
    int index;
    for (index = 0; index < kWarmUpTurns; ++index) {
        cudaMemcpy(pointers[0], pointers[1], kWarmUpSize, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(begin);
    cudaMemcpy(pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, begin, end);
    elapsed /= 1000;

    cudaSetDevice(device_id0);
    // cudaDeviceDisablePeerAccess(device_id1);
    cudaFree(pointers[0]);

    cudaSetDevice(device_id1);
    // cudaDeviceDisablePeerAccess(device_id0);
    cudaFree(pointers[1]);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return elapsed;
}


float P2PCopyTest(int device_id0, int device_id1, size_t size) {
    int *pointer0;
    int *pointer1;

    cudaSetDevice(device_id0);
    cudaMalloc(&pointer0, size);

    cudaSetDevice(device_id1);
    cudaMalloc(&pointer1, size);
    
    // Set device back to device0
    cudaSetDevice(device_id0);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // Check available
    // int avail;
    // cudaDeviceCanAccessPeer(&avail, device_id0, device_id1);
    // printf("[%d]", avail);

    // Warm Up
    int index;
    for (index = 0; index < kWarmUpTurns; ++index) {
        // cudaMemcpyPeer(pointer0, device_id0, pointer1, device_id1, kWarmUpSize);
        cudaMemcpyPeer(pointer1, device_id1, pointer0, device_id0, kWarmUpSize);
        // cudaMemcpyPeer(pointer1, pointer0, kWarmUpSize, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(begin);
    // cudaEventSynchronize(begin);
    cudaMemcpyPeer(pointer0, device_id0, pointer1, device_id1, size);
    cudaDeviceSynchronize();
    // cudaMemcpy(pointers[1], pointers[0], size, cudaMemcpyDeviceToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, begin, end);

    cudaSetDevice(device_id0);
    cudaFree(pointer0);

    cudaSetDevice(device_id1);
    cudaFree(pointer1);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return elapsed;
}

float PinCopyTest(int device_id0, int device_id1, size_t size) {
    int *pointer_device0;
    int *pointer_device1;
    int *pointer_host;

    cudaHostAlloc((void**)&pointer_host, size, cudaHostAllocDefault);
    cudaSetDevice(device_id0);
    cudaMalloc((void**)&pointer_device0, size);
    cudaSetDevice(device_id1);
    cudaMalloc((void**)&pointer_device1, size);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // Warm Up
    cudaEventRecord(begin);
    // cudaEventSynchronize(begin);
    cudaMemcpy(pointer_host, pointer_device0, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pointer_device1, pointer_host, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // cudaMemcpy(pointers[1], pointers[0], size, cudaMemcpyDeviceToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, begin, end);

    cudaFreeHost(pointer_host);

    cudaSetDevice(device_id0);
    cudaFree(pointer_device0);
    cudaSetDevice(device_id1);
    cudaFree(pointer_device1);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return elapsed;
}

int main() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    if (numGPUs < 2) {
        std::cout << "Error: At least two GPUs are required." << std::endl;
        return 0;
    }
    

    printf("Bandwidth test result:\n");

    for (int index1 = 0; index1 < numGPUs; index1++) {
        cudaSetDevice(index1);
        for (int index2 = 0; index2 < numGPUs; index2++) {
            cudaDeviceEnablePeerAccess(index2, 0);
        }
    }
    
    for (int index1 = 0; index1 < numGPUs; index1++) {
        for (int index2 = 0; index2 < numGPUs; index2++) {
            size_t data_size = 128 * 1024 * 1024 * sizeof(int);
            float time = P2PCopyTest(index1, index2, data_size);
            printf("%10.2f", time);
        }
        printf("\n");
    }

    printf("\n");
    printf("Pin memory copy test:\n");
    for (int index1 = 0; index1 < numGPUs; index1++) {
        for (int index2 = 0; index2 < numGPUs; index2++) {
            size_t data_size = 128 * 1024 * 1024 * sizeof(int);
            float time = PinCopyTest(index1, index2, data_size);
            printf("%10.2f", time);
        }
        printf("\n");
    }

    return 0;
}
