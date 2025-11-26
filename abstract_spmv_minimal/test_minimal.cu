// Minimal test to verify compilation
// This is a simplified version that should compile cleanly

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <utility>

using int32 = std::int32_t;
using uint32 = std::uint32_t;

namespace config {
    constexpr uint32 warp_size = 32;
}

constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;

template <typename T>
__host__ __device__ constexpr T zero() { return T{}; }

template <typename T>
__host__ __device__ T ceildivT(T nom, T denom) {
    return (nom + denom - 1ll) / denom;
}

namespace acc {

template <typename ValueType, typename IndexType>
class simple_1d {
public:
    using value_type = ValueType;
    using storage_type = ValueType;
    using arithmetic_type = ValueType;

    __host__ __device__ simple_1d(ValueType* data) : data_(data) {}

    __host__ __device__ ValueType operator()(IndexType idx) const {
        return data_[idx];
    }

private:
    ValueType* data_;
};

template <typename ValueType, typename IndexType>
class simple_2d {
public:
    using value_type = ValueType;
    using storage_type = ValueType;
    using arithmetic_type = ValueType;

    __host__ __device__ simple_2d(IndexType rows, IndexType cols,
                                   ValueType* data, IndexType stride)
        : rows_(rows), cols_(cols), data_(data), stride_(stride) {}

    __host__ __device__ ValueType& operator()(IndexType row, IndexType col) const {
        return data_[row * stride_ + col];
    }

    __host__ __device__ ValueType* get_storage_address(IndexType row, IndexType col) const {
        return &data_[row * stride_ + col];
    }

private:
    IndexType rows_, cols_;
    ValueType* data_;
    IndexType stride_;
};

template <typename Accessor>
class range {
public:
    using value_type = typename Accessor::value_type;
    using storage_type = typename Accessor::storage_type;
    using arithmetic_type = typename Accessor::arithmetic_type;

    __host__ __device__ range(const Accessor& acc) : accessor_(acc) {}

    template <typename... Args>
    __host__ __device__ auto operator()(Args&&... args) const
        -> decltype(std::declval<const Accessor&>()(std::forward<Args>(args)...)) {
        return accessor_(std::forward<Args>(args)...);
    }

    __host__ __device__ const Accessor* operator->() const {
        return &accessor_;
    }

private:
    Accessor accessor_;
};

}  // namespace acc

__global__ void test_kernel(acc::range<acc::simple_1d<double, int32>> vals,
                           acc::range<acc::simple_2d<double, int32>> mat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        printf("Test kernel executed successfully!\n");
    }
}

int main() {
    printf("Minimal compilation test\n");

    double* d_vals;
    double* d_mat;

    cudaMalloc(&d_vals, 10 * sizeof(double));
    cudaMalloc(&d_mat, 100 * sizeof(double));

    acc::simple_1d<double, int32> val_acc(d_vals);
    acc::simple_2d<double, int32> mat_acc(10, 10, d_mat, 10);

    acc::range<acc::simple_1d<double, int32>> val_range(val_acc);
    acc::range<acc::simple_2d<double, int32>> mat_range(mat_acc);

    test_kernel<<<1, 32>>>(val_range, mat_range);
    cudaDeviceSynchronize();

    cudaFree(d_vals);
    cudaFree(d_mat);

    printf("Test completed successfully!\n");
    return 0;
}
