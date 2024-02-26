// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"


#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


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
               uninitialized_array<ValueType, sg_size*(sg_size + 1)>& space)
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
void transpose(
    const size_type nrows, const size_type ncols,
    const ValueType* __restrict__ in, const size_type in_stride,
    ValueType* __restrict__ out, const size_type out_stride,
    sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, DeviceConfig::subgroup_size*(
                                       DeviceConfig::subgroup_size + 1)>& space)
{
    transpose<DeviceConfig::subgroup_size>(
        nrows, ncols, in, in_stride, out, out_stride,
        [](ValueType val) { return val; }, item_ct1, space);
}

template <typename DeviceConfig, typename ValueType>
void transpose(sycl::queue* queue, const matrix::Dense<ValueType>* orig,
               matrix::Dense<ValueType>* trans)
{
    auto size = orig->get_size();
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    dim3 grid(ceildiv(size[1], sg_size), ceildiv(size[0], sg_size));
    dim3 block(sg_size, sg_size);

    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<uninitialized_array<ValueType, sg_size*(sg_size + 1)>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            space_acc_ct1(cgh);
        // Can not pass the member to device function directly
        auto in = orig->get_const_values();
        auto in_stride = orig->get_stride();
        auto out = trans->get_values();
        auto out_stride = trans->get_stride();
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                transpose<DeviceConfig>(size[0], size[1], in, in_stride, out,
                                        out_stride, item_ct1,
                                        *space_acc_ct1.get_pointer());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TYPE(transpose, transpose)
GKO_ENABLE_DEFAULT_CONFIG_CALL_TYPE(transpose_call, transpose);


template <typename DeviceConfig, typename ValueType>
void conj_transpose(
    const size_type nrows, const size_type ncols,
    const ValueType* __restrict__ in, const size_type in_stride,
    ValueType* __restrict__ out, const size_type out_stride,
    sycl::nd_item<3> item_ct1,
    uninitialized_array<ValueType, DeviceConfig::subgroup_size*(
                                       DeviceConfig::subgroup_size + 1)>& space)
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
        sycl::accessor<uninitialized_array<ValueType, sg_size*(sg_size + 1)>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            space_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) __WG_BOUND__(sg_size, sg_size) {
                conj_transpose<DeviceConfig>(nrows, ncols, in, in_stride, out,
                                             out_stride, item_ct1,
                                             *space_acc_ct1.get_pointer());
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
                          const matrix::Dense<ValueType>* x,
                          const matrix::Dense<ValueType>* y,
                          matrix::Dense<ValueType>* result, array<char>& tmp)
{
    // TODO Add onemkl for single column ?
    compute_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_conj_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                               const matrix::Dense<ValueType>* x,
                               const matrix::Dense<ValueType>* y,
                               matrix::Dense<ValueType>* result,
                               array<char>& tmp)
{
    // TODO Add onemkl for single column ?
    compute_conj_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm2_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Dense<ValueType>* x,
                            matrix::Dense<remove_complex<ValueType>>* result,
                            array<char>& tmp)
{
    // TODO Add onemkl for single column ?
    compute_norm2(exec, x, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL);


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* a,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* c)
{
    using namespace oneapi::mkl;
    if (b->get_stride() != 0 && c->get_stride() != 0) {
        if (a->get_size()[1] > 0) {
            oneapi::mkl::blas::row_major::gemm(
                *exec->get_queue(), transpose::nontrans, transpose::nontrans,
                c->get_size()[0], c->get_size()[1], a->get_size()[1],
                one<ValueType>(), a->get_const_values(), a->get_stride(),
                b->get_const_values(), b->get_stride(), zero<ValueType>(),
                c->get_values(), c->get_stride());
        } else {
            dense::fill(exec, c, zero<ValueType>());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* a, const matrix::Dense<ValueType>* b,
           const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* c)
{
    using namespace oneapi::mkl;
    if (b->get_stride() != 0 && c->get_stride() != 0) {
        if (a->get_size()[1] > 0) {
            oneapi::mkl::blas::row_major::gemm(
                *exec->get_queue(), transpose::nontrans, transpose::nontrans,
                c->get_size()[0], c->get_size()[1], a->get_size()[1],
                exec->copy_val_to_host(alpha->get_const_values()),
                a->get_const_values(), a->get_stride(), b->get_const_values(),
                b->get_stride(),
                exec->copy_val_to_host(beta->get_const_values()),
                c->get_values(), c->get_stride());
        } else {
            dense::scale(exec, beta, c);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    const int64* row_ptrs,
                    matrix::Coo<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto in_vals = source->get_const_values();
    const auto stride = source->get_stride();

    auto rows = result->get_row_idxs();
    auto cols = result->get_col_idxs();
    auto vals = result->get_values();

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_rows, [=](sycl::item<1> item) {
            const auto row = static_cast<size_type>(item[0]);
            auto write_to = row_ptrs[row];

            for (size_type col = 0; col < num_cols; col++) {
                if (is_nonzero(in_vals[stride * row + col])) {
                    vals[write_to] = in_vals[stride * row + col];
                    cols[write_to] = static_cast<IndexType>(col);
                    rows[write_to] = static_cast<IndexType>(row);
                    write_to++;
                }
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto in_vals = source->get_const_values();
    const auto stride = source->get_stride();

    const auto row_ptrs = result->get_const_row_ptrs();
    auto cols = result->get_col_idxs();
    auto vals = result->get_values();

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_rows, [=](sycl::item<1> item) {
            const auto row = static_cast<size_type>(item[0]);
            auto write_to = row_ptrs[row];

            for (size_type col = 0; col < num_cols; col++) {
                if (is_nonzero(in_vals[stride * row + col])) {
                    vals[write_to] = in_vals[stride * row + col];
                    cols[write_to] = static_cast<IndexType>(col);
                    write_to++;
                }
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Ell<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto max_nnz_per_row = result->get_num_stored_elements_per_row();
    const auto in_vals = source->get_const_values();
    const auto in_stride = source->get_stride();

    auto cols = result->get_col_idxs();
    auto vals = result->get_values();
    const auto stride = result->get_stride();

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_rows, [=](sycl::item<1> item) {
            const auto row = static_cast<size_type>(item[0]);
            size_type col_idx = 0;
            for (size_type col = 0; col < num_cols; col++) {
                if (is_nonzero(in_vals[row * in_stride + col])) {
                    cols[col_idx * stride + row] = col;
                    vals[col_idx * stride + row] =
                        in_vals[row * in_stride + col];
                    col_idx++;
                }
            }
            for (; col_idx < max_nnz_per_row; col_idx++) {
                cols[col_idx * stride + row] = invalid_index<IndexType>();
                vals[col_idx * stride + row] = zero<ValueType>();
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Dense<ValueType>* source,
                      matrix::Fbcsr<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzero_blocks_per_row(std::shared_ptr<const DefaultExecutor> exec,
                                  const matrix::Dense<ValueType>* source,
                                  int bs,
                                  IndexType* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType>* source,
                       const int64* coo_row_ptrs,
                       matrix::Hybrid<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto ell_lim = result->get_ell_num_stored_elements_per_row();
    const auto in_vals = source->get_const_values();
    const auto in_stride = source->get_stride();
    const auto ell_stride = result->get_ell_stride();
    auto ell_cols = result->get_ell_col_idxs();
    auto ell_vals = result->get_ell_values();
    auto coo_rows = result->get_coo_row_idxs();
    auto coo_cols = result->get_coo_col_idxs();
    auto coo_vals = result->get_coo_values();

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_rows, [=](sycl::item<1> item) {
            const auto row = static_cast<size_type>(item[0]);
            size_type ell_count = 0;
            size_type col = 0;
            auto ell_idx = row;
            for (; col < num_cols && ell_count < ell_lim; col++) {
                const auto val = in_vals[row * in_stride + col];
                if (is_nonzero(val)) {
                    ell_vals[ell_idx] = val;
                    ell_cols[ell_idx] = static_cast<IndexType>(col);
                    ell_count++;
                    ell_idx += ell_stride;
                }
            }
            for (; ell_count < ell_lim; ell_count++) {
                ell_vals[ell_idx] = zero<ValueType>();
                ell_cols[ell_idx] = invalid_index<IndexType>();
                ell_idx += ell_stride;
            }
            auto coo_idx = coo_row_ptrs[row];
            for (; col < num_cols; col++) {
                const auto val = in_vals[row * in_stride + col];
                if (is_nonzero(val)) {
                    coo_vals[coo_idx] = val;
                    coo_cols[coo_idx] = static_cast<IndexType>(col);
                    coo_rows[coo_idx] = static_cast<IndexType>(row);
                    coo_idx++;
                }
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Dense<ValueType>* source,
                      matrix::Sellp<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = source->get_stride();
    const auto in_vals = source->get_const_values();

    const auto slice_sets = result->get_const_slice_sets();
    const auto slice_size = result->get_slice_size();
    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_rows, [=](sycl::item<1> item) {
            const auto row = static_cast<size_type>(item[0]);
            const auto local_row = row % slice_size;
            const auto slice = row / slice_size;
            const auto slice_end = slice_sets[slice + 1] * slice_size;
            auto out_idx = slice_sets[slice] * slice_size + local_row;

            for (size_type col = 0; col < num_cols; col++) {
                const auto val = in_vals[row * stride + col];
                if (is_nonzero(val)) {
                    col_idxs[out_idx] = static_cast<IndexType>(col);
                    vals[out_idx] = val;
                    out_idx += slice_size;
                }
            }
            for (; out_idx < slice_end; out_idx += slice_size) {
                col_idxs[out_idx] = invalid_index<IndexType>();
                vals[out_idx] = zero<ValueType>();
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Dense<ValueType>* source,
                             matrix::SparsityCsr<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto in_vals = source->get_const_values();
    const auto stride = source->get_stride();

    const auto row_ptrs = result->get_const_row_ptrs();
    auto cols = result->get_col_idxs();

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_rows, [=](sycl::item<1> item) {
            const auto row = static_cast<size_type>(item[0]);
            auto write_to = row_ptrs[row];

            for (size_type col = 0; col < num_cols; col++) {
                if (is_nonzero(in_vals[stride * row + col])) {
                    cols[write_to] = static_cast<IndexType>(col);
                    write_to++;
                }
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* orig,
               matrix::Dense<ValueType>* trans)
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* orig,
                    matrix::Dense<ValueType>* trans)
{
    auto size = orig->get_size();
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
                                orig->get_const_values(), orig->get_stride(),
                                trans->get_values(), trans->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
