// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/eigensolver/lobpcg_kernels.hpp"

#include <limits>

#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/dev_lapack_bindings.hpp"


#if GKO_HAVE_LAPACK


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace lobpcg {


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


}  // namespace lobpcg
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

#endif  // GKO_HAVE_LAPACK
