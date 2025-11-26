// Syntax check wrapper for CUDA code using g++
// This won't compile to working code, but will catch syntax errors

#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <utility>

// Mock CUDA types
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

struct cudaError_t {
    int value;
    operator int() const { return value; }
};

const cudaError_t cudaSuccess = {0};

inline const char* cudaGetErrorString(cudaError_t) { return "mock"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaMalloc(void**, size_t) { return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return cudaSuccess; }
inline cudaError_t cudaMemset(void*, int, size_t) { return cudaSuccess; }

enum { cudaMemcpyHostToDevice = 0, cudaMemcpyDeviceToHost = 1 };

namespace cooperative_groups {
    struct thread_block {
        void sync() const {}
        unsigned size() const { return 128; }
        unsigned thread_rank() const { return 0; }
    };

    template <unsigned Size, typename Parent = void>
    struct thread_block_tile {
        void sync() const {}
        unsigned size() const { return Size; }
        unsigned thread_rank() const { return 0; }
        template <typename T>
        T shfl(T var, int) const { return var; }
        template <typename T>
        T shfl_up(T var, unsigned) const { return var; }
        template <typename T>
        T shfl_down(T var, unsigned) const { return var; }
        int any(int) const { return 0; }
        int all(int) const { return 0; }
        unsigned ballot(int) const { return 0; }
    };

    inline thread_block this_thread_block() { return thread_block{}; }

    template <unsigned Size>
    inline thread_block_tile<Size, void> tiled_partition(const thread_block&) {
        return thread_block_tile<Size, void>{};
    }
}

inline double atomicAdd(double* addr, double val) {
    double old = *addr;
    *addr += val;
    return old;
}

// Mock grid variables
struct { unsigned x, y, z; } blockIdx, blockDim, threadIdx, gridDim;

// Now include the type definitions from our code
using size_type = std::int64_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

namespace config {
    constexpr uint32 warp_size = 32;
}

constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;

template <typename T>
__host__ __device__ __forceinline__ constexpr T zero()
{
    return T{};
}

template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}

template <typename T>
__host__ __device__ __forceinline__ T min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__host__ __device__ __forceinline__ T max(T a, T b)
{
    return a > b ? a : b;
}

__forceinline__ __device__ double atomic_add(double* __restrict__ addr, double val)
{
    return atomicAdd(addr, val);
}

namespace group {
    using cooperative_groups::thread_block;
    using cooperative_groups::thread_block_tile;
    using cooperative_groups::this_thread_block;
    using cooperative_groups::tiled_partition;
}

namespace acc {

template <typename ValueType, typename IndexType>
class simple_row_major_2d {
public:
    using value_type = ValueType;
    using storage_type = ValueType;
    using arithmetic_type = ValueType;

    __host__ __device__ simple_row_major_2d(
        IndexType num_rows, IndexType num_cols,
        ValueType* data, IndexType stride)
        : num_rows_(num_rows), num_cols_(num_cols),
          data_(data), stride_(stride)
    {}

    __host__ __device__ ValueType& operator()(IndexType row, IndexType col) const
    {
        return data_[row * stride_ + col];
    }

    __host__ __device__ ValueType* get_storage_address(IndexType row, IndexType col) const
    {
        return &data_[row * stride_ + col];
    }

    __host__ __device__ IndexType length(int dimension) const
    {
        return dimension == 0 ? num_rows_ : num_cols_;
    }

private:
    IndexType num_rows_;
    IndexType num_cols_;
    ValueType* data_;
    IndexType stride_;
};

template <typename ValueType, typename IndexType>
class simple_1d {
public:
    using value_type = ValueType;
    using storage_type = ValueType;
    using arithmetic_type = ValueType;

    __host__ __device__ simple_1d(ValueType* data)
        : data_(data)
    {}

    __host__ __device__ ValueType operator()(IndexType idx) const
    {
        return data_[idx];
    }

private:
    ValueType* data_;
};

template <typename Accessor>
class range {
public:
    using accessor_type = Accessor;
    using value_type = typename Accessor::value_type;
    using storage_type = typename Accessor::storage_type;
    using arithmetic_type = typename Accessor::arithmetic_type;

    __host__ __device__ range(const Accessor& acc) : accessor_(acc) {}

    template <typename... Args>
    __host__ __device__ auto operator()(Args... args) const
        -> decltype(std::declval<Accessor>()(args...))
    {
        return accessor_(args...);
    }

    __host__ __device__ const Accessor* operator->() const
    {
        return &accessor_;
    }

    template <typename T = Accessor>
    __host__ __device__ auto length(int dimension) const
        -> decltype(std::declval<T>().length(dimension))
    {
        return accessor_.length(dimension);
    }

private:
    Accessor accessor_;
};

}  // namespace acc

int main() {
    printf("Syntax check passed!\n");
    return 0;
}
