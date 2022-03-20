
#include <cuda_runtime.h>
#include <time.h>
#include <vector>

__global__ void slowKernel1(float* output, bool write = false)
{
    float input = 5;
#pragma unroll
    for (int i = 0; i < 10000000; i++) {
        input = input * .9999999;
    }
    if (write) *output -= input;
}

__global__ void slowKernel2(float* output, bool write = false)
{
    float input = 5;
#pragma unroll
    for (int i = 0; i < 10000000; i++) {
        input = input * .9999999;
    }
    if (write) *output -= input;
}

__global__ void fastKernel1(float* output, bool write = false)
{
    float input = 5;
#pragma unroll
    for (int i = 0; i < 100000; i++) {
        input = input * .9999999;
    }
    if (write) *output -= input;
}

__global__ void fastKernel2(float* output, bool write = false)
{
    float input = 5;
#pragma unroll
    for (int i = 0; i < 100000; i++) {
        input = input * .9999999;
    }
    if (write) *output -= input;
}

void burntime(long val)
{
    struct timespec tv[] = {{0, val}};
    nanosleep(tv, 0);
}


void mallocSynchronize()
{
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
    const size_t sz = 1 << 10;
    float a1 = 1.5;
    slowKernel1<<<1, 1, 0, stream1>>>((float*)(0));
    burntime(500000000L);  // 500ms wait - slowKernel around 1300ms
    int* dev_a1 = 0;
    int* dev_a2 = 0;
    int* dev_a3 = 0;
    float* da1 = 0;
    float* da2 = 0;
    cudaMalloc(&da1, 1);
    cudaMalloc(&da2, 1);
    for (int i = 0; i < 10; i++) {
        cudaMemcpyAsync((void*)da1, (void*)(&a1), sizeof(float),
                        cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync((void*)da2, (void*)(&a1), sizeof(float),
                        cudaMemcpyHostToDevice, stream2);
        slowKernel2<<<1, 1, 0, stream1>>>(da1, false);
        // cudaFree(dev_a1);
        // cudaMalloc(&dev_a2, sz);
        slowKernel1<<<1, 1, 0, stream1>>>(da1, false);
        cudaStreamSynchronize(stream1);
        slowKernel2<<<1, 1, 0, stream2>>>(da2, false);
        // cudaFree(dev_a1);
        // cudaMalloc(&dev_a2, sz);
        slowKernel1<<<1, 1, 0, stream2>>>(da2, false);
        cudaStreamSynchronize(stream2);
        // cudaFree(dev_a2);
        // cudaMalloc(&dev_a3, sz);
        // slowKernel1<<<1, 1, 0, stream3>>>((float*)(0));
        burntime(1000000L);  // 1ms wait - fastKernel around 15ms
    }
    cudaFree(dev_a1);
}

int main()
{
    mallocSynchronize();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
