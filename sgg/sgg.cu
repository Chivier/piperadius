#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

float P2PCopyTest(int device_id1, int device_id2, size_t size) {
    int *pointers[2];

    cudaSetDevice(device_id1);
    cudaDeviceEnablePeerAccess(device_id2, 0);
    cudaMalloc(&pointers[0], size);

    cudaSetDevice(device_id2);
    cudaDeviceEnablePeerAccess(device_id1, 0);
    cudaMalloc(&pointers[1], size);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    cudaEventRecord(begin);
    cudaMemcpyAsync(pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, begin, end);
    elapsed /= 1000;

    cudaSetDevice(device_id1);
    cudaFree(pointers[0]);

    cudaSetDevice(device_id2);
    cudaFree(pointers[1]);

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

    for (int index1 = 0; index1 < numGPUs; index1++) {
        for (int index2 = 0; index2 < numGPUs; index2++) {
            size_t data_size = 64 * 1024 * 1024;
            float time = P2PCopyTest(index1, index2, data_size);
            float bandwidth = (data_size / 1024 / 1024) / (time); // MB/s
            printf("%10f", bandwidth * bandwidth);
        }
        printf("\n");
    }

    return 0;
}
