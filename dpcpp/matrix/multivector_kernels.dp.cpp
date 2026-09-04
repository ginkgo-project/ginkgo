// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/multivector_kernels.hpp"

#include <oneapi/mkl.hpp>

#include <sycl/sycl.hpp>

#include <ginkgo/core/base/math.hpp>

#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/base/math.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/base/types.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace multivector {


// Disable the 64 subgroup. CPU supports 64 now, but conj_transpose will
// lead CL_OUT_OF_RESOURCES. TODO: investigate this issue.
constexpr auto dcfg_1d_list = dcfg_1d_list_t();
constexpr auto subgroup_list = dcfg_1sg_list_t();
constexpr auto dcfg_sq_list = dcfg_sq_list_t();
constexpr auto dcfg_1d_array = syn::as_array(dcfg_1d_list);
constexpr int default_block_size = 256;


namespace kernel {


template <std::uint32_t sg_size, typename ValueType, typename Closure>
void transpose(const size_type nrows, const size_type ncols,
               const ValueType* __restrict__ in, const size_type in_stride,
               ValueType* __restrict__ out, const size_type out_stride,
               Closure op, sycl::nd_item<3> item_ct1,
               sycl::local_accessor<device_type<ValueType>, 1> space)
{
    auto local_x = item_ct1.get_local_id(2);
    auto local_y = item_ct1.get_local_id(1);
    auto x = item_ct1.get_group(2) * sg_size + local_x;
    auto y = item_ct1.get_group(1) * sg_size + local_y;
    if (y < nrows && x < ncols) {
        space[local_y * (sg_size + 1) + local_x] = op(in[y * in_stride + x]);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    x = item_ct1.get_group(1) * sg_size + local_x;
    y = item_ct1.get_group(2) * sg_size + local_y;
    if (y < ncols && x < nrows) {
        out[y * out_stride + x] = space[local_x * (sg_size + 1) + local_y];
    }
}

template <typename DeviceConfig, typename ValueType>
void transpose(const size_type nrows, const size_type ncols,
               const ValueType* __restrict__ in, const size_type in_stride,
               ValueType* __restrict__ out, const size_type out_stride,
               sycl::nd_item<3> item_ct1,
               sycl::local_accessor<device_type<ValueType>, 1> space)
{
    transpose<DeviceConfig::subgroup_size>(
        nrows, ncols, in, in_stride, out, out_stride,
        [](ValueType val) { return val; }, item_ct1, space);
}

template <typename DeviceConfig, typename ValueType>
void transpose(sycl::queue* queue, matrix::view::dense<const ValueType> orig,
               matrix::view::dense<ValueType> trans)
{
    auto size = orig.size;
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    dim3 grid(ceildiv(size[1], sg_size), ceildiv(size[0], sg_size));
    dim3 block(sg_size, sg_size);

    queue->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<device_type<ValueType>, 1> space_acc(
            sg_size * (sg_size + 1), cgh);
        // Can not pass the member to device function directly
        auto in = as_device_type(orig.values);
        auto in_stride = orig.stride;
        auto out = as_device_type(trans.values);
        auto out_stride = trans.stride;
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                transpose<DeviceConfig>(size[0], size[1], in, in_stride, out,
                                        out_stride, item_ct1, space_acc);
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TYPE(transpose, transpose)
GKO_ENABLE_DEFAULT_CONFIG_CALL_TYPE(transpose_call, transpose);


template <typename DeviceConfig, typename ValueType>
void conj_transpose(const size_type nrows, const size_type ncols,
                    const ValueType* __restrict__ in, const size_type in_stride,
                    ValueType* __restrict__ out, const size_type out_stride,
                    sycl::nd_item<3> item_ct1,
                    sycl::local_accessor<device_type<ValueType>, 1> space)
{
    transpose<DeviceConfig::subgroup_size>(
        nrows, ncols, in, in_stride, out, out_stride,
        [](ValueType val) { return conj(val); }, item_ct1, space);
}

template <typename DeviceConfig, typename ValueType>
void conj_transpose(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                    sycl::queue* queue, const size_type nrows,
                    const size_type ncols, const ValueType* in,
                    const size_type in_stride, ValueType* out,
                    const size_type out_stride)
{
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    queue->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<device_type<ValueType>, 1> space_acc(
            sg_size * (sg_size + 1), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) __WG_BOUND__(sg_size, sg_size) {
                conj_transpose<DeviceConfig>(nrows, ncols, in, in_stride, out,
                                             out_stride, item_ct1, space_acc);
            });
    });
}


GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(conj_transpose,
                                                  conj_transpose, DCFG_1D);
GKO_ENABLE_DEFAULT_CONFIG_CALL(conj_transpose_call, conj_transpose,
                               dcfg_sq_list);


}  // namespace kernel


template <typename ValueType>
void compute_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                          matrix::view::dense<const ValueType> x,
                          matrix::view::dense<const ValueType> y,
                          matrix::view::dense<ValueType> result,
                          array<char>& tmp)
{
    // TODO Add onemkl for single column ?
    compute_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_conj_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                               matrix::view::dense<const ValueType> x,
                               matrix::view::dense<const ValueType> y,
                               matrix::view::dense<ValueType> result,
                               array<char>& tmp)
{
    // TODO Add onemkl for single column ?
    compute_conj_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               matrix::view::dense<const ValueType> orig,
               matrix::view::dense<ValueType> trans)
{
    auto queue = exec->get_queue();
    kernel::transpose_call(
        dcfg_sq_type_list_t(),
        [&queue](auto cfg) {
            const auto sg_size = cfg.subgroup_size;
            return validate(queue, cfg.block_size, sg_size) &&
                   sg_size * (sg_size + 1) * sizeof(ValueType) <=
                       queue->get_device()
                           .get_info<sycl::info::device::local_mem_size>();
        },
        queue, orig, trans);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,
                    matrix::view::dense<const ValueType> orig,
                    matrix::view::dense<ValueType> trans)
{
    auto size = orig.size;
    auto sq_array = syn::as_array(dcfg_sq_list);
    auto queue = exec->get_queue();
    const std::uint32_t cfg =
        get_first_cfg(sq_array, [&queue](std::uint32_t cfg) {
            const auto sg_size = DCFG_1D::decode<1>(cfg);
            return validate(queue, DCFG_1D::decode<0>(cfg), sg_size) &&
                   sg_size * (sg_size + 1) * sizeof(ValueType) <=
                       queue->get_device()
                           .get_info<sycl::info::device::local_mem_size>();
        });
    const auto sg_size = DCFG_1D::decode<1>(cfg);
    dim3 grid(ceildiv(size[1], sg_size), ceildiv(size[0], sg_size));
    dim3 block(sg_size, sg_size);
    kernel::conj_transpose_call(cfg, grid, block, 0, queue, size[0], size[1],
                                as_device_type(orig.values), orig.stride,
                                as_device_type(trans.values), trans.stride);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_CONJ_TRANSPOSE_KERNEL);


}  // namespace multivector
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
