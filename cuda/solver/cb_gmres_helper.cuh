#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>


#include "accessor/reduced_row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {


namespace cb_gmres {


// Specialization, so the Accessor can use the same function as regular pointers
template <int dim, typename Type1, typename Type2>
GKO_INLINE auto as_cuda_accessor(
    const acc::range<acc::reduced_row_major<dim, Type1, Type2>> &acc)
{
    return acc::range<
        acc::reduced_row_major<dim, cuda_type<Type1>, cuda_type<Type2>>>(
        acc.get_accessor().get_size(),
        as_cuda_type(acc.get_accessor().get_stored_data()),
        acc.get_accessor().get_stride());
}

template <int dim, typename Type1, typename Type2, size_type mask>
GKO_INLINE auto as_cuda_accessor(
    const acc::range<acc::scaled_reduced_row_major<dim, Type1, Type2, mask>>
        &acc)
{
    return acc::range<acc::scaled_reduced_row_major<dim, cuda_type<Type1>,
                                                    cuda_type<Type2>, mask>>(
        acc.get_accessor().get_size(),
        as_cuda_type(acc.get_accessor().get_stored_data()),
        acc.get_accessor().get_storage_stride(),
        as_cuda_type(acc.get_accessor().get_scalar()),
        acc.get_accessor().get_scalar_stride());
}


}  // namespace cb_gmres


class OrthStorage {
private:
    using storage_type = double;  // Make sure all data is properly aligned
    // Number of storage_type elems
    static constexpr size_type default_num_elems{128 * 128};

public:
    template <typename T>
    static T *get_data(const std::shared_ptr<const Executor> &exec,
                       size_type num_elems)
    {
        auto &obj = get_obj(exec);
        if (obj->storage_size_ < num_elems * sizeof(T)) {
            obj->allocate(exec.get(),
                          (num_elems * sizeof(T) / sizeof(storage_type)) + 1);
        }

        return reinterpret_cast<T *>(obj->data_);
    }

private:
    OrthStorage(const std::shared_ptr<const Executor> &exec)
        : exec_{exec.get()}, storage_size_{0}, data_{nullptr}
    {}
    // The data stored in here is not freed intentionally because the executor
    // could have already expired. If the executor was stored in a shared_ptr,
    // we get in trouble

    void allocate(const Executor *exec, size_type num_elems)
    {
        singleton_->data_ = exec->template alloc<storage_type>(num_elems);
        singleton_->storage_size_ = num_elems * sizeof(storage_type);
    }

    void free(const Executor *exec) { exec->free(data_); }

    static std::unique_ptr<OrthStorage> &get_obj(
        const std::shared_ptr<const Executor> &exec)
    {
        if (!singleton_) {
            singleton_.reset(new OrthStorage(exec));
        } else if (singleton_->exec_ != exec.get()) {
            // WARNING: This creates a data leak for old data
            singleton_->allocate(exec.get(), default_num_elems);
        }
        return singleton_;
    }

    static std::unique_ptr<OrthStorage> singleton_;

    // plain pointer to not delay the deletion of the executors
    const Executor *exec_;
    size_type storage_size_;  // storage size in bytes
    storage_type *data_;
};

std::unique_ptr<OrthStorage> OrthStorage::singleton_{};


// ValueType is non-complex type
template <int shared_size, typename Accessor3d, typename ValueType>
__global__ void compute_dot_norm(size_type num_vectors, Accessor3d krylov_bases,
                                 ValueType *__restrict__ output)
{
    using c_value_type = typename Accessor3d::accessor::arithmetic_type;
    if (blockIdx.x > 0 || threadIdx.x >= num_vectors) {
        return;
    }
    ValueType result[shared_size];
    if (threadIdx.x < shared_size) {
        result[threadIdx.x] = zero<ValueType>();
    }

    auto tblock = group::this_thread_block();
    auto warp = group::tiled_partition<config::warp_size>(tblock);
    const size_type num_warps = (blockDim.x - 1) / config::warp_size + 1;
    const size_type start_i = num_vectors % num_warps;

    for (size_type k = warp.thread_rank(); k < krylov_bases.length(1);
         k += config::warp_size) {
        for (size_type i = start_i; i < num_vectors; i += num_warps) {
            const auto v1 = krylov_bases(i, k, 0);
            for (size_type j = 0; j < num_vectors; ++j) {
                const auto local_result =
                    squared_norm(v1 * krylov_bases(j, k, 0));
                const auto reduced_result =
                    reduce(warp, local_result,
                           [](ValueType a, ValueType b) { return a + b; });
                if (warp.thread_rank() == 0) {
                    result[j] += reduced_result;
                }
            }
        }
    }
    for (int k = shared_size / 2; k >= config::warp_size; k /= 2) {
        tblock.sync();
        if (threadIdx.x + k < shared_size) {
            result[threadIdx.x] = result[threadIdx.x] + result[threadIdx.x + k];
        }
    }
    if (threadIdx.x < config::warp_size) {
        const auto final_res =
            reduce(warp, result[threadIdx.x],
                   [](ValueType a, ValueType b) { return a + b; });
        if (threadIdx.x == 0) {
            *output = final_res;
        }
    }
}

template <typename T, typename Accessor3d>
T get_orthogonality(std::shared_ptr<const CudaExecutor> &exec,
                    size_type num_vectors, Accessor3d krylov_bases, T *d_tmp)
{
    constexpr int shared_size{128};
    constexpr int block_size{shared_size};
    const dim3 block_dot_norm(block_size, 1, 1);
    compute_dot_norm<shared_size><<<1, block_dot_norm>>>(
        num_vectors, cb_gmres::as_cuda_accessor(krylov_bases->to_const()),
        as_cuda_type(d_tmp));
    return sqrt(exec->copy_val_to_host(d_tmp));
}


// TODO: compute ||-V^T * V|| to show loss of orthogonality
template <bool before, typename ValueType>
__global__ void print_norms(const size_type krylov_dim,
                            const ValueType *__restrict__ arnoldi_norm,
                            const size_type norm_stride)
{
    constexpr remove_complex<ValueType> eta_squared = 1.0 / 2.0;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ValueType num0;
        ValueType num11;
        for (size_type i = 0; i < krylov_dim; ++i) {
            if (before) {
                num0 = sqrt(eta_squared * arnoldi_norm[i]);
                num11 = sqrt(arnoldi_norm[i + norm_stride]);
            } else {
                num0 = arnoldi_norm[i];
                num11 = arnoldi_norm[i + norm_stride];
            }
            printf("%f,%f  ", num0, num11);
        }
    }
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko
