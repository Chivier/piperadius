#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#define THREADS_PER_BLOCK 256  // Number of threads per block
#define MAX_BLOCKS 1024        // Maximum number of thread blocks
#define MIN_CHUNK_SIZE 4       // Use 4 bytes (1 int) as the smallest chunk size

// Kernel to simulate random read/write access
__global__ void RandomAccessKernel(int *data, int *indexes, size_t chunk_count, size_t chunk_size) {
    // Shared memory to simulate cache-restricted computation
    extern __shared__ int shared[];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread ID

    if (tid < chunk_count) {
        // Read a random chunk index for the thread
        size_t start_idx = indexes[tid] * chunk_size;

        // Read chunk into shared memory
        for (size_t i = 0; i < chunk_size; i++) {
            shared[threadIdx.x + i] = data[start_idx + i];
        }

        __syncthreads();

        // Perform some computation on the data (to prevent compiler optimization)
        for (size_t i = 0; i < chunk_size; i++) {
            shared[threadIdx.x + i] += 1;
        }

        __syncthreads();

        // Write the result back to global memory
        for (size_t i = 0; i < chunk_size; i++) {
            data[start_idx + i] = shared[threadIdx.x + i];
        }
    }
}

// Host function to generate random indexes for the kernel
void generate_random_indexes(int *host_indexes, size_t range, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, range - 1);

    for (size_t i = 0; i < size; i++) {
        host_indexes[i] = distrib(gen);  // Random chunk index
    }
}

// Host function to measure kernel bandwidth
float measure_kernel_bandwidth(size_t chunk_size, size_t chunk_count) {
    // Set the size of total data
    size_t mem_size = chunk_size * chunk_count * sizeof(int);

    // Allocate device memory
    int *d_data, *d_indexes;
    cudaMalloc(&d_data, mem_size);
    cudaMalloc(&d_indexes, chunk_count * sizeof(int));

    // Create random indexes for chunks
    std::vector<int> h_indexes(chunk_count);
    generate_random_indexes(h_indexes.data(), chunk_count, chunk_count);

    // Transfer data and indexes to device
    cudaMemcpy(d_indexes, h_indexes.data(), chunk_count * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads_per_block(THREADS_PER_BLOCK);
    dim3 blocks_per_grid((chunk_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up kernel to mitigate driver overhead
    RandomAccessKernel<<<blocks_per_grid, threads_per_block, THREADS_PER_BLOCK * sizeof(int)>>>(d_data, d_indexes, chunk_count, chunk_size);
    cudaDeviceSynchronize();

    // Launch kernel and record timestamps
    cudaEventRecord(start);
    RandomAccessKernel<<<blocks_per_grid, threads_per_block, THREADS_PER_BLOCK * sizeof(int)>>>(d_data, d_indexes, chunk_count, chunk_size);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Clean up memory
    cudaFree(d_data);
    cudaFree(d_indexes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate bandwidth: (bytes transferred)/(time taken)
    float bandwidth = (float(mem_size) / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0);
    return bandwidth;
}

int main() {
    std::cout << "GPU Memory Bandwidth with Cache Effects:\n";
    std::cout << "Chunk Size (Bytes)\tBandwidth (GB/s)\n";

    size_t max_chunk_size = 1024 * 1024;  // Experiment with chunks up to 1 MB
    size_t chunk_count = 1024 * 64;      // Number of chunks (use large datasets)

    for (size_t chunk_size = MIN_CHUNK_SIZE; chunk_size <= max_chunk_size; chunk_size *= 2) {
        float bandwidth = measure_kernel_bandwidth(chunk_size, chunk_count);
        std::cout << chunk_size << "\t" << bandwidth << "\n";
    }

    return 0;
}

