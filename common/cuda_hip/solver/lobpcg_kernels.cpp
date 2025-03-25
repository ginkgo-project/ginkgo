// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/lobpcg_kernels.hpp"

#include <limits>

#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/dev_lapack_bindings.hpp"

#if GKO_HAVE_LAPACK


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace lobpcg {


using gko::kernels::lobpcg::workspace_mode;


template <typename ValueType>
void symm_generalized_eig(std::shared_ptr<const DefaultExecutor> exec,
                          const workspace_mode alloc,
                          matrix::Dense<ValueType>* a,
                          matrix::Dense<ValueType>* b,
                          array<remove_complex<ValueType>>* e_vals,
                          array<char>* workspace)
{
    const auto id = exec->get_device_id();
    auto handle = exec->get_dev_lapack_handle();

    constexpr auto max = std::numeric_limits<int>::max();
    if (a->get_size()[1] > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int)));
    }
    if (a->get_stride() > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int)));
    }
    if (b->get_stride() > max) {
        throw OverflowError(__FILE__, __LINE__,
                            name_demangling::get_type_name(typeid(int)));
    }
    int n = static_cast<int>(a->get_size()[1]);  // column-major
    int lda = static_cast<int>(a->get_stride());
    int ldb = static_cast<int>(b->get_stride());
    int fp_buffer_num_elems;
    if (alloc == workspace_mode::allocate) {
        dev_lapack::sygvd_buffersize(handle, LAPACK_EIG_TYPE_1,
                                     LAPACK_EIG_VECTOR, LAPACK_FILL_LOWER, n,
                                     a->get_values(), lda, b->get_values(), ldb,
                                     e_vals->get_data(), &fp_buffer_num_elems);
        size_type total_bytes = sizeof(ValueType) * fp_buffer_num_elems;
        workspace->resize_and_reset(total_bytes);
    } else {
        fp_buffer_num_elems = workspace->get_size() / sizeof(ValueType);
    }
    array<int> dev_info(exec, 1);
    try {
        dev_lapack::sygvd(handle, LAPACK_EIG_TYPE_1, LAPACK_EIG_VECTOR,
                          LAPACK_FILL_LOWER, n, a->get_values(), lda,
                          b->get_values(), ldb, e_vals->get_data(),
                          reinterpret_cast<ValueType*>(workspace->get_data()),
                          fp_buffer_num_elems, dev_info.get_data());
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        int host_info = exec->copy_val_to_host(dev_info.get_data());
        std::cout << "devInfo was " << host_info << std::endl;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_LOBPCG_SYMM_GENERALIZED_EIG_KERNEL);


}  // namespace lobpcg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HAVE_LAPACK
