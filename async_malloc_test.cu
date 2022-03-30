#include <cuda_runtime.h>

#include <vector>


__global__ void kernel(float* data, int size)
{
#pragma unroll
    for (int i = 0; i < size; i++) {
        data[i] = .5 + i;
    }
}


void free_synchronize()
{
    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    const size_t length = 1 << 10;
    float* dev_a1 = 0;
    float* dev_a2 = 0;
    float* a1;
    float* a2;
    // std::vector<float> a2(length, 0.0);
    // std::vector<float> a1(length, 0.0);
    cudaHostAlloc(&a1, sizeof(float) * length, 0);
    cudaHostAlloc(&a2, sizeof(float) * length, 0);

    cudaMalloc(&dev_a1, length);
    for (int i = 0; i < 10; i++) {
        kernel<<<1, 1, 0, stream1>>>(dev_a1, length);
        cudaMemcpyAsync(a1, dev_a1, sizeof(float), cudaMemcpyDefault, stream1);
        a1[i] += 1.5;
        cudaMemcpyAsync(dev_a1, a1, sizeof(float), cudaMemcpyDefault, stream1);
    }
    cudaMalloc(&dev_a2, length);
    for (int i = 0; i < 10; i++) {
        kernel<<<1, 1, 0, stream2>>>(dev_a2, length);
        cudaMemcpyAsync(a2, dev_a2, sizeof(float), cudaMemcpyDeviceToHost,
                        stream2);
        a2[i] += 1.5;
        cudaMemcpyAsync(dev_a2, a2, sizeof(float), cudaMemcpyDefault, stream2);
    }
    cudaFree(dev_a1);
    cudaFree(dev_a2);
    cudaFreeHost(a1);
    cudaFreeHost(a2);
}

int main()
{
    free_synchronize();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
