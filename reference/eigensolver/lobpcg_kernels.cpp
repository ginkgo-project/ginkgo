// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/eigensolver/lobpcg_kernels.hpp"

#include <ginkgo/core/base/types.hpp>

#include "reference/base/blas_bindings.hpp"
#include "reference/base/lapack_bindings.hpp"

#if GKO_HAVE_LAPACK


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The LOBPCG solver namespace.
 *
 * @ingroup lobpcg
 */
namespace lobpcg {


template <typename ValueType>
void symm_eig(std::shared_ptr<const ReferenceExecutor> exec,
              matrix::Dense<ValueType>* a,
              array<remove_complex<ValueType>>* e_vals, array<char>* workspace)
{
    constexpr auto max = std::numeric_limits<int32>::max();
    if (a->get_size()[1] > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    if (a->get_stride() > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int32)));
    }
    const int32 n = static_cast<int32>(a->get_size()[0]);
    const int32 lda = static_cast<int32>(a->get_stride());
    const char job = LAPACK_EIG_VECTOR;
    const char uplo = LAPACK_FILL_LOWER;

    if constexpr (!gko::is_complex_s<ValueType>::value) {
        // Even if the workspace is already allocated, we need to know where to
        // set the pointers for the individual workspaces of LAPACK
        int32 fp_buffer_num_elems;
        int32 int_buffer_num_elems;
        ValueType* work = reinterpret_cast<ValueType*>(workspace->get_data());
        array<int32> tmp_iwork(exec, 1);
        lapack::syevd_buffersizes(
            &job, &uplo, &n, a->get_values(), &lda, e_vals->get_data(), work,
            &fp_buffer_num_elems, tmp_iwork.get_data(), &int_buffer_num_elems);
        size_type total_bytes = sizeof(ValueType) * fp_buffer_num_elems +
                                sizeof(int32) * int_buffer_num_elems;
        if (workspace->get_size() < total_bytes) {
            workspace->resize_and_reset(total_bytes);
        }
        work = reinterpret_cast<ValueType*>(workspace->get_data());
        // Set iwork pointer inside the workspace array
        int32* iwork = reinterpret_cast<int32*>(
            workspace->get_data() + sizeof(ValueType) * fp_buffer_num_elems);
        lapack::syevd(&job, &uplo, &n, a->get_values(), &lda,
                      e_vals->get_data(), work, &fp_buffer_num_elems, iwork,
                      &int_buffer_num_elems);
    } else {  // Complex data type
        int32 fp_buffer_num_elems;
        int32 rfp_buffer_num_elems;
        int32 int_buffer_num_elems;
        ValueType* work = reinterpret_cast<ValueType*>(workspace->get_data());
        array<int32> tmp_iwork(exec, 1);
        array<remove_complex<ValueType>> tmp_rwork(exec, 1);
        lapack::heevd_buffersizes(
            &job, &uplo, &n, a->get_values(), &lda, e_vals->get_data(), work,
            &fp_buffer_num_elems, tmp_rwork.get_data(), &rfp_buffer_num_elems,
            tmp_iwork.get_data(), &int_buffer_num_elems);
        size_type total_bytes =
            sizeof(ValueType) * fp_buffer_num_elems +
            sizeof(remove_complex<ValueType>) * rfp_buffer_num_elems +
            sizeof(int32) * int_buffer_num_elems;
        if (workspace->get_size() < total_bytes) {
            workspace->resize_and_reset(total_bytes);
        }
        work = reinterpret_cast<ValueType*>(workspace->get_data());
        // Set rwork and iwork pointers inside the workspace array
        remove_complex<ValueType>* rwork =
            reinterpret_cast<remove_complex<ValueType>*>(
                workspace->get_data() +
                sizeof(ValueType) * fp_buffer_num_elems);
        int32* iwork = reinterpret_cast<int32*>(
            workspace->get_data() + sizeof(ValueType) * fp_buffer_num_elems +
            sizeof(remove_complex<ValueType>) * rfp_buffer_num_elems);
        lapack::heevd(&job, &uplo, &n, a->get_values(), &lda,
                      e_vals->get_data(), work, &fp_buffer_num_elems, rwork,
                      &rfp_buffer_num_elems, iwork, &int_buffer_num_elems);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_LOBPCG_SYMM_EIG_KERNEL);


template <typename ValueType>
void symm_generalized_eig(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::Dense<ValueType>* a,
                          matrix::Dense<ValueType>* b,
                          array<remove_complex<ValueType>>* e_vals,
                          array<char>* workspace)
{
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
    const int32 n = static_cast<int32>(a->get_size()[0]);
    const int32 lda = static_cast<int32>(a->get_stride());
    const int32 ldb = static_cast<int32>(b->get_stride());
    const int32 itype = 1;
    const char job = LAPACK_EIG_VECTOR;
    const char uplo = LAPACK_FILL_LOWER;

    if constexpr (!gko::is_complex_s<ValueType>::value) {
        // Even if the workspace is already allocated, we need to know where to
        // set the pointers for the individual workspaces of LAPACK
        int32 fp_buffer_num_elems;
        int32 int_buffer_num_elems;
        ValueType* work = reinterpret_cast<ValueType*>(workspace->get_data());
        array<int32> tmp_iwork(exec, 1);
        lapack::sygvd_buffersizes(
            &itype, &job, &uplo, &n, a->get_values(), &lda, b->get_values(),
            &ldb, e_vals->get_data(), work, &fp_buffer_num_elems,
            tmp_iwork.get_data(), &int_buffer_num_elems);
        size_type total_bytes = sizeof(ValueType) * fp_buffer_num_elems +
                                sizeof(int32) * int_buffer_num_elems;
        if (workspace->get_size() < total_bytes) {
            workspace->resize_and_reset(total_bytes);
        }
        work = reinterpret_cast<ValueType*>(workspace->get_data());
        // Set iwork pointer inside the workspace array
        int32* iwork = reinterpret_cast<int32*>(
            workspace->get_data() + sizeof(ValueType) * fp_buffer_num_elems);
        lapack::sygvd(&itype, &job, &uplo, &n, a->get_values(), &lda,
                      b->get_values(), &ldb, e_vals->get_data(), work,
                      &fp_buffer_num_elems, iwork, &int_buffer_num_elems);
    } else {  // Complex data type
        int32 fp_buffer_num_elems;
        int32 rfp_buffer_num_elems;
        int32 int_buffer_num_elems;
        ValueType* work = reinterpret_cast<ValueType*>(workspace->get_data());
        array<int32> tmp_iwork(exec, 1);
        array<remove_complex<ValueType>> tmp_rwork(exec, 1);
        lapack::hegvd_buffersizes(
            &itype, &job, &uplo, &n, a->get_values(), &lda, b->get_values(),
            &ldb, e_vals->get_data(), work, &fp_buffer_num_elems,
            tmp_rwork.get_data(), &rfp_buffer_num_elems, tmp_iwork.get_data(),
            &int_buffer_num_elems);
        size_type total_bytes =
            sizeof(ValueType) * fp_buffer_num_elems +
            sizeof(remove_complex<ValueType>) * rfp_buffer_num_elems +
            sizeof(int32) * int_buffer_num_elems;
        if (workspace->get_size() < total_bytes) {
            workspace->resize_and_reset(total_bytes);
        }
        work = reinterpret_cast<ValueType*>(workspace->get_data());
        // Set rwork and iwork pointers inside the workspace array
        remove_complex<ValueType>* rwork =
            reinterpret_cast<remove_complex<ValueType>*>(
                workspace->get_data() +
                sizeof(ValueType) * fp_buffer_num_elems);
        int32* iwork = reinterpret_cast<int32*>(
            workspace->get_data() + sizeof(ValueType) * fp_buffer_num_elems +
            sizeof(remove_complex<ValueType>) * rfp_buffer_num_elems);
        lapack::hegvd(&itype, &job, &uplo, &n, a->get_values(), &lda,
                      b->get_values(), &ldb, e_vals->get_data(), work,
                      &fp_buffer_num_elems, rwork, &rfp_buffer_num_elems, iwork,
                      &int_buffer_num_elems);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_LOBPCG_SYMM_GENERALIZED_EIG_KERNEL);


template <typename ValueType>
void b_orthonormalize(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::Dense<ValueType>* a, LinOp* b,
                      array<char>* workspace)  // (unused; for [cu/hip]SOLVER)
{
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

    // Cholesky
    // Since LAPACK expects column-major, on exit, we will have
    // L such that LL^H = A^T, i.e., the complex conjugate of the
    // lower Cholesky factor, in column-major order.
    const char uplo = LAPACK_FILL_LOWER;
    lapack::potrf(&uplo, &n, aH_b_a->get_values(), &ldaH_b_a);

    // Invert the Cholesky factor: on exit, have conj(L)^{-1}
    const char diag = LAPACK_DIAG_NONUNIT;
    lapack::trtri(&uplo, &diag, &n, aH_b_a->get_values(), &ldaH_b_a);

    // A = A * (L^{-1})^H
    // Since A is seen by BLAS as column-major, the operation
    // A^T_{ij} = M_{ik} A^T_{kj}, with M = conj(L)^{-1} (col-major),
    // is equivalent to A_{ji} = A_{jk} M^T_{ki} =
    // A = A * L^{-H} (in row-major order).
    const char side = BLAS_SIDE_LEFT;
    const ValueType alpha = gko::one<ValueType>();
    const char transa = BLAS_OP_N;
    const int32 m = static_cast<int32>(a->get_size()[0]);
    // m & n swapped because of interpreting as col-major
    blas::trmm(&side, &uplo, &transa, &diag, &n, &m, &alpha,
               aH_b_a->get_const_values(), &ldaH_b_a, a->get_values(), &lda);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_LOBPCG_B_ORTHONORMALIZE_KERNEL);


}  // namespace lobpcg
}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HAVE_LAPACK
