#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

const int kWarmUpTurns = 10;
const size_t kWarmUpSize = 8 * 1024 * 1024; // 8 MB

float DirectCopyTest(int device_id1, int device_id2, size_t size) {
    int *pointers[2];

    cudaSetDevice(device_id1);
    // cudaDeviceEnablePeerAccess(device_id2, 0);
    cudaMalloc(&pointers[0], size);

    cudaSetDevice(device_id2);
    // cudaDeviceEnablePeerAccess(device_id1, 0);
    cudaMalloc(&pointers[1], size);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // Check available
    // int avail;
    // cudaDeviceCanAccessPeer(&avail, device_id1, device_id2);
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

    cudaSetDevice(device_id1);
    // cudaDeviceDisablePeerAccess(device_id2);
    cudaFree(pointers[0]);

    cudaSetDevice(device_id2);
    // cudaDeviceDisablePeerAccess(device_id1);
    cudaFree(pointers[1]);

    cudaEventDestroy(end);
    cudaEventDestroy(begin);

    return elapsed;
}


float P2PCopyTest(int device_id1, int device_id2, size_t size) {
    int *pointer0;
    int *pointer1;

    cudaSetDevice(device_id1);
    cudaMalloc(&pointer0, size);
    // cudaDeviceEnablePeerAccess(device_id2, 0);

    cudaSetDevice(device_id2);
    cudaMalloc(&pointer1, size);
    // cudaDeviceEnablePeerAccess(device_id1, 0);

    cudaSetDevice(device_id1);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // Check available
    // int avail;
    // cudaDeviceCanAccessPeer(&avail, device_id1, device_id2);
    // printf("[%d]", avail);

    // Warm Up
    int index;
    for (index = 0; index < kWarmUpTurns; ++index) {
        // cudaMemcpyPeer(pointer0, device_id1, pointer1, device_id2, kWarmUpSize);
        cudaMemcpyPeer(pointer1, device_id2, pointer0, device_id1, kWarmUpSize);
        // cudaMemcpyPeer(pointer1, pointer0, kWarmUpSize, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(begin);
    // cudaEventSynchronize(begin);
    cudaMemcpyPeer(pointer0, device_id1, pointer1, device_id2, size);
    cudaDeviceSynchronize();
    // cudaMemcpy(pointers[1], pointers[0], size, cudaMemcpyDeviceToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, begin, end);
    elapsed /= 1000;

    cudaSetDevice(device_id1);
    cudaFree(pointer0);

    cudaSetDevice(device_id2);
    cudaFree(pointer1);

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
            float bandwidth = (data_size / 1024 / 1024 / 1024.0) / (time); // GB/s
            printf("%10.2f,", bandwidth);
        }
        printf("\n");
    }
    /*
    printf("Bandwidth test result:\n");

    for (int index1 = 0; index1 < numGPUs; index1++) {
        for (int index2 = 0; index2 < numGPUs; index2++) {
            size_t data_size = 128 * 1024 * 1024 * sizeof(int);
            float time = DirectCopyTest(index1, index2, data_size);
            float bandwidth = (data_size / 1024 / 1024 / 1024.0) / (time); // GB/s
            printf("%10.2f,", bandwidth * bandwidth);
        }
        printf("\n");
    }
    */

    return 0;
}
