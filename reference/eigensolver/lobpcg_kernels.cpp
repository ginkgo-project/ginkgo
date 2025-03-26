// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/eigensolver/lobpcg_kernels.hpp"

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
    const int32 n = static_cast<int32>(a->get_size()[1]);  // column-major
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
        if (alloc == workspace_mode::allocate) {
            size_type total_bytes = sizeof(ValueType) * fp_buffer_num_elems +
                                    sizeof(int32) * int_buffer_num_elems;
            workspace->resize_and_reset(total_bytes);
            work = reinterpret_cast<ValueType*>(workspace->get_data());
        }
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
        if (alloc == workspace_mode::allocate) {
            size_type total_bytes =
                sizeof(ValueType) * fp_buffer_num_elems +
                sizeof(remove_complex<ValueType>) * rfp_buffer_num_elems +
                sizeof(int32) * int_buffer_num_elems;
            workspace->resize_and_reset(total_bytes);
            work = reinterpret_cast<ValueType*>(workspace->get_data());
        }
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


}  // namespace lobpcg
}  // namespace reference
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HAVE_LAPACK
