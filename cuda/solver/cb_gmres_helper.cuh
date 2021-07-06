/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ios>
#include <iostream>
#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>


#include "accessor/reduced_row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {

/* TODO
 * copy the next_krylov_vector to krylov_bases before computing the
   orthogonality
 * Optimize orthogonalization process
 * Measure orthogonality only up until the restart (100 iterations)

*/
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


class stream_guard {
public:
    stream_guard(std::ostream &out)
        : stream{out}, old_flags{out.flags()}, old_state{nullptr}
    {
        old_state.copyfmt(out);
    }
    ~stream_guard()
    {
        stream.copyfmt(old_state);
        stream.flags(old_flags);
    }
    /*
// save the old state of flags / settings of cout
std::ios old_state(nullptr);
old_state.copyfmt(std::cout);
std::ios::fmtflags flags(std::cout.flags());
// ...
std::cout.copyfmt(old_state);
std::cout.flags(flags);

       */

private:
    std::ostream &stream;
    std::ios::fmtflags old_flags;
    std::ios old_state;
};


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType, typename Accessor3d>
__global__ __launch_bounds__(block_size) void copy_next_krylov_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ next_krylov_basis, size_type stride_next_krylov,
    Accessor3d krylov_bases, const ValueType *__restrict__ hessenberg_iter,
    size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status)
{
    const auto global_id = thread::get_thread_id_flat();
    const auto row_idx = global_id / stride_next_krylov;
    const auto col_idx = global_id % stride_next_krylov;
    const auto hessenberg =
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_next_krylov + col_idx;

        const auto next_krylov_value =
            next_krylov_basis[next_krylov_idx] / hessenberg;

        // next_krylov_basis[next_krylov_idx] = next_krylov_value;
        krylov_bases(iter + 1, row_idx, col_idx) = next_krylov_value;
    }
}

// ValueType is non-complex type
template <int shared_size, typename Accessor3d, typename ValueType>
__global__ void compute_dot_norm(size_type num_vectors, Accessor3d krylov_bases,
                                 ValueType *__restrict__ output)
{
    using c_value_type = typename Accessor3d::accessor::arithmetic_type;
    if (blockIdx.x > 0) {
        return;
    }
    __shared__ ValueType result[shared_size];
    if (threadIdx.x < shared_size) {
        result[threadIdx.x] = zero<ValueType>();
    }

    auto tblock = group::this_thread_block();
    auto warp = group::tiled_partition<config::warp_size>(tblock);
    const size_type num_warps = (blockDim.x - 1) / config::warp_size + 1;
    const size_type start_i = threadIdx.x / config::warp_size;

    const size_type k_end = krylov_bases.length(1) + config::warp_size -
                            krylov_bases.length(1) % config::warp_size;

    if (threadIdx.x == 0) printf("%lluS ", num_vectors);
    for (size_type k = warp.thread_rank(); k <= k_end; k += config::warp_size) {
        for (size_type i = start_i; i < num_vectors; i += num_warps) {
            const auto v1 =
                k < num_vectors ? krylov_bases(i, k, 0) : ValueType{};
            for (size_type j = 0; j < num_vectors; ++j) {
                const auto v2 =
                    k < num_vectors ? krylov_bases(j, k, 0) : ValueType{};
                const auto local_result = squared_norm(v1 * v2);
                const auto reduced_result =
                    reduce(warp, local_result,
                           [](ValueType a, ValueType b) { return a + b; });
                if (warp.thread_rank() == 0) {
                    result[j] += reduced_result;
                }
            }
        }
    }
    if (threadIdx.x == 0) printf("%lluM ", num_vectors);
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
    if (threadIdx.x == 0) printf("%lluE ", num_vectors);
}

// ValueType is non-complex type
template <int block_size, typename Accessor3d, typename ValueType>
__global__ void compute_dot_matrix_atomic(size_type num_vectors,
                                          Accessor3d krylov_bases,
                                          ValueType *__restrict__ output)
{
    GKO_ASSERT(block_size == blockDim.x);
    using value_type = typename Accessor3d::accessor::arithmetic_type;
    static_assert(std::is_same<value_type, ValueType>::value,
                  "Types must match!");
    // using nc_value_type =
    //    remove_complex<typename Accessor3d::accessor::arithmetic_type>;
    auto tblock = group::this_thread_block();
    __shared__ UninitializedArray<value_type, block_size> result_ar;
    auto result = static_cast<value_type *>(result_ar);

    const size_type vector_length{krylov_bases.length(1)};

    const size_type chunk_size = vector_length / gridDim.x + 1;
    const size_type slice_start = blockIdx.x * chunk_size;
    const size_type slice_end_tmp = (blockIdx.x + 1) * chunk_size;
    const size_type slice_end =
        vector_length < slice_end_tmp ? vector_length : slice_end_tmp;
    for (size_type i = 0; i < num_vectors; ++i) {
        for (size_type j = i; j < num_vectors; ++j) {
            value_type local_res = 0;
            for (size_type k = slice_start + threadIdx.x; k < slice_end;
                 k += block_size) {
                const value_type d =
                    krylov_bases(i, k, 0) * krylov_bases(j, k, 0);
                local_res += d;  // squared_norm(d);
            }
            result[threadIdx.x] = local_res;
            reduce(tblock, result,
                   [](value_type a, value_type b) { return a + b; });
            if (threadIdx.x == 0) {
                atomic_add(&output[i * num_vectors + j], result[0]);
            }
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // printf("<%llu, %llu // %u>", slice_start, slice_end, gridDim.x);
    }
}

template <int block_size, typename ValueType, typename NcValueType>
__global__ __launch_bounds__(block_size) void compute_matrix_norm(
    size_type N, const ValueType *__restrict__ mtx,
    NcValueType *__restrict__ norm)
{
    GKO_ASSERT(block_size == blockDim.x);
    auto tblock = group::this_thread_block();

    __shared__ NcValueType result[block_size];
    NcValueType local_res{};
    for (size_type i = blockIdx.x; i < N; i += gridDim.x) {
        for (size_type j = i + 1 + threadIdx.x; j < N; j += block_size) {
            const auto mtx_val = mtx[i * N + j];
            local_res += squared_norm(mtx_val);
        }
    }
    result[threadIdx.x] = local_res;
    tblock.sync();
    reduce(tblock, result, [](NcValueType a, NcValueType b) { return a + b; });
    if (threadIdx.x == 0) {
        atomic_add(norm, result[0]);
    }
}


template <typename T, typename Accessor3d>
remove_complex<T> get_orthogonality(std::shared_ptr<const CudaExecutor> &exec,
                                    size_type num_vectors,
                                    Accessor3d krylov_bases, T *d_tmp)
{
    using value_type = T;
    using nc_value_type = remove_complex<T>;
    // constexpr int block_size{32};  // 256};
    constexpr int block_size{256};

    const size_type num_vec_elems = krylov_bases.length(1);

    auto d_norm =
        reinterpret_cast<nc_value_type *>(d_tmp + num_vectors * num_vectors);
    components::fill_array(exec, d_tmp, num_vectors * num_vectors + 1,
                           zero<value_type>());
    const int sms = exec->get_num_multiprocessor();
    const int grid_dot = std::min<int>(4 * sms, num_vec_elems);
    const int grid_norm = std::min<int>(4 * sms, num_vectors);
    compute_dot_matrix_atomic<block_size><<<grid_dot, block_size>>>(
        num_vectors, cb_gmres::as_cuda_accessor(krylov_bases->to_const()),
        as_cuda_type(d_tmp));
    compute_matrix_norm<block_size>
        <<<grid_norm, block_size>>>(num_vectors, as_cuda_type(d_tmp), d_norm);
    const auto norm_result = sqrt(exec->copy_val_to_host(d_norm));

    // Reference:
#if false
    //*
    stream_guard guard{std::cout};
    std::cout.precision(17);
    std::cout << std::scientific;

    size_type num_elems = num_vectors * num_vec_elems;
    using ar_type = value_type;
    // std::remove_const_t<typename Accessor3d::accessor::arithmetic_type>;
    using st_type =
        std::remove_const_t<typename Accessor3d::accessor::storage_type>;
    std::vector<st_type> krylov_vectors(num_elems);
    std::vector<value_type> mm_res(num_vectors * num_vectors);

    nc_value_type ref_norm{};
    exec->synchronize();

    exec->get_master()->copy_from(exec.get(), num_elems,
                                  krylov_bases->get_const_storage(),
                                  krylov_vectors.data());
    exec->get_master()->copy_from(exec.get(), num_vectors * num_vectors, d_tmp,
                                  mm_res.data());

   // std::cout << "Iteration GPU: " << num_vectors << '\n';
   // for (size_type i = 0; i < num_vectors; ++i) {
   //     for (size_type j = 0; j < num_vectors; ++j) {
   //         const ar_type v1 = mm_res[i * num_vectors + j];
   //         std::cout << v1 << '\t';
   //     }
   //     std::cout << '\n';
   // }
    //std::cout << "Iteration CPU: " << num_vectors << '\n';
    for (size_type i = 0; i < num_vectors; ++i) {
        for (size_type abc = 0; abc < i; ++abc) {
        //    std::cout << 0.0 << '\t';
        }
        for (size_type j = i; j < num_vectors; ++j) {
            auto local_res = zero<value_type>();
            for (size_type k = 0; k < num_vec_elems; ++k) {
                const ar_type v1 = krylov_vectors[i * num_vec_elems + k];
                const ar_type v2 = krylov_vectors[j * num_vec_elems + k];
                local_res += v1 * v2;
            }
         //   std::cout << local_res << '\t';
            if (i != j) {
                ref_norm += squared_norm(local_res);
            }
        }
        //std::cout << '\n';
    }
    /*
    std::cout << "Iteration: " << num_vectors << '\n';
    for (size_type k = 0; k < num_vec_elems; ++k) {
        for (size_type i = 0; i < num_vectors; ++i) {
            const ar_type v1 = krylov_vectors[i * num_vec_elems + k];
            std::cout << v1 << '\t';
        }
        std::cout << '\n';
    }
    //*/
    ref_norm = sqrt(ref_norm);
    const auto max = std::max(ref_norm, norm_result);
    const T diff = (max == 0) ? T{0} : (ref_norm - norm_result) / max;
    //std::cout << '{' << norm_result << "<->" << ref_norm << '_' << diff
    //          << "}";
    return ref_norm;
#endif
    return norm_result;
    /*
    constexpr int shared_size{128};
    const dim3 block_dot_norm(block_size, 1, 1);
    compute_dot_norm<shared_size><<<1, block_dot_norm>>>(
        num_vectors, cb_gmres::as_cuda_accessor(krylov_bases->to_const()),
        as_cuda_type(d_tmp));
    return sqrt(exec->copy_val_to_host(d_tmp));
    */
}

/*
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
*/


}  // namespace cuda
}  // namespace kernels
}  // namespace gko
