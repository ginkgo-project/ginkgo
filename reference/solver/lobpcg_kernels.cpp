// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/lobpcg_kernels.hpp"

#include <ginkgo/core/base/types.hpp>

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


using gko::kernels::lobpcg::workspace_mode;

template <typename ValueType>
void symm_generalized_eig(std::shared_ptr<const ReferenceExecutor> exec,
                          const workspace_mode alloc,
                          matrix::Dense<ValueType>* a,
                          matrix::Dense<ValueType>* b,
                          array<remove_complex<ValueType>>* e_vals,
                          array<char>* workspace)
{
    const char job = LAPACK_EIG_VECTOR;
    const char uplo = LAPACK_FILL_LOWER;
    const int n = static_cast<int>(a->get_size()[1]);
    const int lda = static_cast<int>(a->get_stride());
    const int ldb = static_cast<int>(b->get_stride());
    const int itype = 1;
    if constexpr (!gko::is_complex_s<ValueType>::value) {
        // Even if the workspace is already allocated, we need to know where to
        // se set the pointers for the individual workspaces of LAPACK
        int fp_buffer_num_elems;
        int int_buffer_num_elems;
        array<int> tmp_iwork(exec, 1);
        lapack::sygvd_buffersizes(
            &itype, &job, &uplo, &n, a->get_values(), &lda, b->get_values(),
            &ldb, e_vals->get_data(), (ValueType*)workspace->get_data(),
            &fp_buffer_num_elems, tmp_iwork.get_data(), &int_buffer_num_elems);
        if (alloc == workspace_mode::allocate) {
            size_type total_bytes = sizeof(ValueType) * fp_buffer_num_elems +
                                    sizeof(int) * int_buffer_num_elems;
            workspace->resize_and_reset(total_bytes);
        }
        // Set work and iwork pointers inside the workspace array
        ValueType* work = (ValueType*)(workspace->get_data());
        int* iwork = (int*)(workspace->get_data() +
                            sizeof(ValueType) * fp_buffer_num_elems);
        lapack::sygvd(&itype, &job, &uplo, &n, a->get_values(), &lda,
                      b->get_values(), &ldb, e_vals->get_data(), work,
                      &fp_buffer_num_elems, iwork, &int_buffer_num_elems);
    } else  // Complex data type
    {
        int fp_buffer_num_elems;
        int rfp_buffer_num_elems;
        int int_buffer_num_elems;
        array<remove_complex<ValueType>> tmp_rwork(exec, 1);
        array<int> tmp_iwork(exec, 1);
        lapack::hegvd_buffersizes(
            &itype, &job, &uplo, &n, a->get_values(), &lda, b->get_values(),
            &ldb, e_vals->get_data(), (ValueType*)workspace->get_data(),
            &fp_buffer_num_elems, tmp_rwork.get_data(), &rfp_buffer_num_elems,
            tmp_iwork.get_data(), &int_buffer_num_elems);
        if (alloc == workspace_mode::allocate) {
            size_type total_bytes =
                sizeof(ValueType) * fp_buffer_num_elems +
                sizeof(remove_complex<ValueType>) * rfp_buffer_num_elems +
                sizeof(int) * int_buffer_num_elems;
            workspace->resize_and_reset(total_bytes);
        }
        // Set work and iwork pointers inside the workspace array
        ValueType* work = (ValueType*)(workspace->get_data());
        remove_complex<ValueType>* rwork =
            (remove_complex<ValueType>*)(workspace->get_data() +
                                         sizeof(ValueType) *
                                             fp_buffer_num_elems);
        int* iwork =
            (int*)(workspace->get_data() +
                   sizeof(ValueType) * fp_buffer_num_elems +
                   sizeof(remove_complex<ValueType>) * rfp_buffer_num_elems);
        lapack::hegvd(&itype, &job, &uplo, &n, a->get_values(), &lda,
                      b->get_values(), &ldb, e_vals->get_data(), work,
                      &fp_buffer_num_elems, rwork, &rfp_buffer_num_elems, iwork,
                      &int_buffer_num_elems);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_LOBPCG_SYMM_GENERALIZED_EIG_KERNEL);


}  // namespace lobpcg
}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HAVE_LAPACK
