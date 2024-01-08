// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/device_matrix_data_kernels.hpp"


#include <ginkgo/core/base/types.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename ValueType, typename IndexType>
void soa_to_aos(std::shared_ptr<const DefaultExecutor> exec,
                const device_matrix_data<ValueType, IndexType>& in,
                array<matrix_data_entry<ValueType, IndexType>>& out)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto rows, auto cols, auto vals, auto out) {
            out[i] = {rows[i], cols[i], vals[i]};
        },
        in.get_num_stored_elements(), in.get_const_row_idxs(),
        in.get_const_col_idxs(), in.get_const_values(), out);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_SOA_TO_AOS_KERNEL);


template <typename ValueType, typename IndexType>
void aos_to_soa(std::shared_ptr<const DefaultExecutor> exec,
                const array<matrix_data_entry<ValueType, IndexType>>& in,
                device_matrix_data<ValueType, IndexType>& out)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto in, auto rows, auto cols, auto vals) {
            rows[i] = in[i].row;
            cols[i] = in[i].column;
            vals[i] = unpack_member(in[i].value);
        },
        in.get_size(), in, out.get_row_idxs(), out.get_col_idxs(),
        out.get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DEVICE_MATRIX_DATA_AOS_TO_SOA_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
