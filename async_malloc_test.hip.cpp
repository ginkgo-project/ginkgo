
#include <hip/hip_runtime.h>
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
    hipStream_t stream1, stream2, stream3;
    hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking);
    hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking);
    hipStreamCreateWithFlags(&stream3, hipStreamNonBlocking);
    const size_t sz = 1 << 10;
    float a1 = 1.5;
    slowKernel1<<<1, 1, 0, stream1>>>((float*)(0));
    burntime(500000000L);  // 500ms wait - slowKernel around 1300ms
    int* dev_a1 = 0;
    int* dev_a2 = 0;
    int* dev_a3 = 0;
    float* da1 = 0;
    float* da2 = 0;
    hipMalloc(&da1, 1);
    hipMalloc(&da2, 1);
    for (int i = 0; i < 10; i++) {
        hipMemcpyAsync((void*)da1, (void*)(&a1), sizeof(float),
                       hipMemcpyHostToDevice, stream1);
        hipMemcpyAsync((void*)da2, (void*)(&a1), sizeof(float),
                       hipMemcpyHostToDevice, stream2);
        hipMalloc(&dev_a1, sz);
        slowKernel2<<<1, 1, 0, stream1>>>(da1, false);
        hipFree(dev_a1);
        // hipStreamSynchronize(stream1);
        hipMalloc(&dev_a2, sz);
        slowKernel2<<<1, 1, 0, stream2>>>(da2, false);
        hipMalloc(&dev_a3, sz);
        hipFree(dev_a2);
        slowKernel1<<<1, 1, 0, stream3>>>(da2, false);
        // hipStreamSynchronize(stream2);
        hipFree(dev_a3);
        // hipMalloc(&dev_a3, sz);
        // slowKernel1<<<1, 1, 0, stream3>>>((float*)(0));
        burntime(1000000L);  // 1ms wait - fastKernel around 15ms
    }
    hipFree(dev_a1);
}

int main()
{
    mallocSynchronize();
    hipDeviceSynchronize();
    hipDeviceReset();
    return 0;
}
