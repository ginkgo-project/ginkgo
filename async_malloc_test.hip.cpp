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
    std::vector<float> a2(length, 0.0);
    std::vector<float> a1(length, 0.0);

    hipMalloc(&dev_a1, length);
    for (int i = 0; i < 10; i++) {
        kernel<<<1, 1, 0, stream1>>>(dev_a1, length);
        // hipMemcpyAsync(a1.data(), dev_a1, sizeof(float),
        //                 hipMemcpyDeviceToHost, stream1);
        a1[i] += 1.5;
        hipMemcpyAsync(dev_a1, a1.data(), sizeof(float), hipMemcpyHostToDevice,
                       stream1);
    }
    hipMalloc(&dev_a2, length);
    for (int i = 0; i < 10; i++) {
        kernel<<<1, 1, 0, stream2>>>(dev_a2, length);
        // hipMemcpyAsync(a2.data(), dev_a2, sizeof(float),
        //                 hipMemcpyDeviceToHost, stream2);
        a2[i] += 1.5;
        hipMemcpyAsync(dev_a2, a2.data(), sizeof(float), hipMemcpyHostToDevice,
                       stream2);
    }
    hipFree(dev_a1);
    hipFree(dev_a2);
}

int main()
{
    free_synchronize();
    hipDeviceSynchronize();
    hipDeviceReset();
    return 0;
}
