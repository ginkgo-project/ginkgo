// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/eigensolver/lobpcg_kernels.hpp"

#include <limits>

#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/blas_bindings.hpp"
#include "common/cuda_hip/base/dev_lapack_bindings.hpp"
#include "common/cuda_hip/base/pointer_mode_guard.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"

#if GKO_HAVE_LAPACK


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace lobpcg {


constexpr int default_block_size = 512;


namespace kernel {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void fill_lower_col_major(
    const int32 n, const ValueType* source, const int32 source_stride,
    ValueType* dest, const int32 dest_stride)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx % n;
    const auto col = tidx / n;
    const ValueType zero = gko::zero<ValueType>();
    if (row < n && col < n) {
        dest[col * dest_stride + row] =
            (row >= col) ? source[col * source_stride + row] : zero;
    }
}


}  // namespace kernel


template <typename ValueType>
void symm_eig(std::shared_ptr<const DefaultExecutor> exec,
              matrix::Dense<ValueType>* a,
              array<remove_complex<ValueType>>* e_vals, array<char>* workspace)
{
    const auto id = exec->get_device_id();
    auto handle = exec->get_dev_lapack_handle();

    constexpr auto max = std::numeric_limits<int32>::max();
    if (a->get_size()[1] > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    if (a->get_stride() > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    int32 n = static_cast<int32>(a->get_size()[1]);  // column-major
    int32 lda = static_cast<int32>(a->get_stride());
    int32 fp_buffer_num_elems;
    dev_lapack::syevd_buffersize(handle, LAPACK_EIG_VECTOR, LAPACK_FILL_LOWER,
                                 n, a->get_values(), lda, e_vals->get_data(),
                                 &fp_buffer_num_elems);
    size_type total_bytes = sizeof(ValueType) * fp_buffer_num_elems;
    if (workspace->get_size() < total_bytes) {
        workspace->resize_and_reset(total_bytes);
    }
    array<int32> dev_info(exec, 1);
    try {
        dev_lapack::syevd(handle, LAPACK_EIG_VECTOR, LAPACK_FILL_LOWER, n,
                          a->get_values(), lda, e_vals->get_data(),
                          reinterpret_cast<ValueType*>(workspace->get_data()),
                          fp_buffer_num_elems, dev_info.get_data());

        int32 host_info = exec->copy_val_to_host(dev_info.get_data());
        if (host_info != 0) {
            throw GKO_CUSOLVER_ERROR(CUSOLVER_STATUS_INTERNAL_ERROR);
        }
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        int32 host_info = exec->copy_val_to_host(dev_info.get_data());
        std::cout << "devInfo was " << host_info << std::endl;
        throw;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_LOBPCG_SYMM_EIG_KERNEL);


template <typename ValueType>
void symm_generalized_eig(std::shared_ptr<const DefaultExecutor> exec,
                          matrix::Dense<ValueType>* a,
                          matrix::Dense<ValueType>* b,
                          array<remove_complex<ValueType>>* e_vals,
                          array<char>* workspace)
{
    const auto id = exec->get_device_id();
    auto handle = exec->get_dev_lapack_handle();

    constexpr auto max = std::numeric_limits<int32>::max();
    if (a->get_size()[1] > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    if (a->get_stride() > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    if (b->get_stride() > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    int32 n = static_cast<int32>(a->get_size()[1]);  // column-major
    int32 lda = static_cast<int32>(a->get_stride());
    int32 ldb = static_cast<int32>(b->get_stride());
    int32 fp_buffer_num_elems;
    dev_lapack::sygvd_buffersize(handle, LAPACK_EIG_TYPE_1, LAPACK_EIG_VECTOR,
                                 LAPACK_FILL_LOWER, n, a->get_values(), lda,
                                 b->get_values(), ldb, e_vals->get_data(),
                                 &fp_buffer_num_elems);
    size_type total_bytes = sizeof(ValueType) * fp_buffer_num_elems;
    if (workspace->get_size() < total_bytes) {
        workspace->resize_and_reset(total_bytes);
    }
    array<int32> dev_info(exec, 1);
    try {
        dev_lapack::sygvd(handle, LAPACK_EIG_TYPE_1, LAPACK_EIG_VECTOR,
                          LAPACK_FILL_LOWER, n, a->get_values(), lda,
                          b->get_values(), ldb, e_vals->get_data(),
                          reinterpret_cast<ValueType*>(workspace->get_data()),
                          fp_buffer_num_elems, dev_info.get_data());

        int32 host_info = exec->copy_val_to_host(dev_info.get_data());
        if (host_info != 0) {
            throw GKO_CUSOLVER_ERROR(CUSOLVER_STATUS_INTERNAL_ERROR);
        }
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        int32 host_info = exec->copy_val_to_host(dev_info.get_data());
        std::cout << "devInfo was " << host_info << std::endl;
        throw;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_LOBPCG_SYMM_GENERALIZED_EIG_KERNEL);


template <typename ValueType>
void b_orthonormalize(std::shared_ptr<const DefaultExecutor> exec,
                      matrix::Dense<ValueType>* a, LinOp* b,
                      array<char>* workspace)
{
    const auto id = exec->get_device_id();
    auto handle = exec->get_dev_lapack_handle();

    constexpr auto max = std::numeric_limits<int32>::max();
    if (a->get_size()[0] > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    if (a->get_stride() > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    const int32 lda = static_cast<int32>(a->get_stride());

    // Compute A^H * B * A
    auto b_a = matrix::Dense<ValueType>::create(
        exec, gko::dim<2>{b->get_size()[0], a->get_size()[1]});
    b->apply(a, b_a);
    auto aH_b_a = matrix::Dense<ValueType>::create(
        exec, gko::dim<2>{a->get_size()[1], a->get_size()[1]});
    gko::as<matrix::Dense<ValueType>>(a->conj_transpose())->apply(b_a, aH_b_a);

    const int32 n = static_cast<int32>(aH_b_a->get_size()[0]);
    const int32 ldaH_b_a = static_cast<int32>(aH_b_a->get_stride());

    // Cholesky factorization
    int32 fp_buffer_num_elems;
    dev_lapack::potrf_buffersize(handle, LAPACK_FILL_LOWER, n,
                                 aH_b_a->get_values(), ldaH_b_a,
                                 &fp_buffer_num_elems);
    size_type total_bytes = sizeof(ValueType) * fp_buffer_num_elems;
    if (workspace->get_size() < total_bytes) {
        workspace->resize_and_reset(total_bytes);
    }
    // LAPACK uses column-major, so using LOWER produces LL^H = A^T:
    // L (col-major) is the complex conjugate of the lower factor for A.
    array<int32> dev_info(exec, 1);
    try {
        dev_lapack::potrf(handle, LAPACK_FILL_LOWER, n, aH_b_a->get_values(),
                          ldaH_b_a,
                          reinterpret_cast<ValueType*>(workspace->get_data()),
                          fp_buffer_num_elems, dev_info.get_data());

        int32 host_info = exec->copy_val_to_host(dev_info.get_data());
        if (host_info != 0) {
            throw GKO_CUSOLVER_ERROR(CUSOLVER_STATUS_INTERNAL_ERROR);
        }
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        int32 host_info = exec->copy_val_to_host(dev_info.get_data());
        std::cout << "error in potrf: devInfo was " << host_info << std::endl;
        throw;
    }

    // Solve with L as the right hand side: LL^H X = L --> X = L^{-H}.
    // Recall L is the complex conjugate of the "true" L, so really
    // we will have (L_true)^{-T}, stored in column-major format, after potrs.
    auto factor = matrix::Dense<ValueType>::create(exec, aH_b_a->get_size());
    const auto grid_dim = ceildiv(n * n, default_block_size);
    if (grid_dim > 0) {
        kernel::fill_lower_col_major<<<grid_dim, default_block_size, 0,
                                       exec->get_stream()>>>(
            n, as_device_type(aH_b_a->get_const_values()), ldaH_b_a,
            as_device_type(factor->get_values()), ldaH_b_a);
    }
    try {
        dev_lapack::potrs(handle, LAPACK_FILL_LOWER, n, factor->get_size()[1],
                          aH_b_a->get_values(), ldaH_b_a, factor->get_values(),
                          ldaH_b_a, dev_info.get_data());

        int32 host_info = exec->copy_val_to_host(dev_info.get_data());
        if (host_info != 0) {
            throw GKO_CUSOLVER_ERROR(CUSOLVER_STATUS_INTERNAL_ERROR);
        }
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        int32 host_info = exec->copy_val_to_host(dev_info.get_data());
        std::cout << "error in potrs: devInfo was " << host_info << std::endl;
        throw;
    }

    // A = A * (L^{-1})^H
    // A will be seen by BLAS as column-major, or A^T. The BLAS operation
    // A^T_{ij} = F^H_{ik} A^T_{kj}, with F being the "factor" variable,
    // is equivalent to A_{ji} = A_{jk} conj(F)_{ki} --> A = A * conj(F).
    // Since F = (L_true)^{-T}, we have A = A * (L_true)^{-H} in row-major
    // storage upon exit.
    auto blas_handle = exec->get_blas_handle();
    blas::pointer_mode_guard pm_guard(blas_handle);
    const ValueType alpha = gko::one<ValueType>();
    const int32 m = static_cast<int32>(a->get_size()[0]);
    if constexpr (!gko::is_complex_s<ValueType>::value) {
        blas::trmm(blas_handle, BLAS_SIDE_LEFT, LAPACK_FILL_UPPER, BLAS_OP_T,
                   BLAS_DIAG_NONUNIT, n, m, &alpha, factor->get_const_values(),
                   ldaH_b_a, a->get_values(), lda, a->get_values(), lda);
    } else {
        blas::trmm(blas_handle, BLAS_SIDE_LEFT, LAPACK_FILL_UPPER, BLAS_OP_C,
                   BLAS_DIAG_NONUNIT, n, m, &alpha, factor->get_const_values(),
                   ldaH_b_a, a->get_values(), lda, a->get_values(), lda);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_LOBPCG_B_ORTHONORMALIZE_KERNEL);


}  // namespace lobpcg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HAVE_LAPACK
