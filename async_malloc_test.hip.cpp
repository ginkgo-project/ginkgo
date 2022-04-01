#include <hip/hip_runtime.h>

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
    hipStream_t stream1, stream2;
    hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking);
    hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking);
    const size_t length = 1 << 10;
    float* dev_a1 = 0;
    float* dev_a2 = 0;
    float* a1;
    float* a2;
    // std::vector<float> a2(length, 0.0);
    // std::vector<float> a1(length, 0.0);
    hipHostMalloc((void**)(&a1), sizeof(float) * length,
                  // hipHostRegisterDefault,
                  hipHostRegisterPortable);

    hipHostMalloc((void**)(&a2), sizeof(float) * length,
                  // hipHostRegisterDefault,
                  hipHostRegisterPortable);

    hipMalloc(&dev_a1, length);
    hipMalloc(&dev_a2, length);
    for (int i = 0; i < 10; i++) {
        kernel<<<1, 1, 0, stream1>>>(dev_a1, length);
        hipMemcpyAsync(a1, dev_a1, sizeof(float), hipMemcpyDefault, stream1);
        a1[i] += 1.5;
        hipMemcpyAsync(dev_a1, a1, sizeof(float), hipMemcpyDefault, stream1);
    }
    for (int i = 0; i < 10; i++) {
        kernel<<<1, 1, 0, stream2>>>(dev_a2, length);
        hipMemcpyAsync(a2, dev_a2, sizeof(float), hipMemcpyDefault, stream2);
        a2[i] += 1.5;
        hipMemcpyAsync(dev_a2, a2, sizeof(float), hipMemcpyDefault, stream2);
    }
    hipFree(dev_a1);
    hipFree(dev_a2);
    hipHostFree(a1);
    hipHostFree(a2);
}

int main()
{
    free_synchronize();
    hipDeviceSynchronize();
    hipDeviceReset();
    return 0;
}
