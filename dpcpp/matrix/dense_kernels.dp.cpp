// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include <oneapi/mkl.hpp>

#include <ginkgo/core/base/math.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/multivector_kernels.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/base/math.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/base/types.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  matrix::view::dense<const ValueType> a,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> c)
{
    using namespace oneapi::mkl;
    if constexpr (onemkl::is_supported<ValueType>::value) {
        if (b.stride != 0 && c.stride != 0) {
            if (a.size[1] > 0 && a.values && b.values && c.values) {
                oneapi::mkl::blas::row_major::gemm(
                    *exec->get_queue(), transpose::nontrans,
                    transpose::nontrans, c.size[0], c.size[1], a.size[1],
                    one<ValueType>(), as_device_type(a.values), a.stride,
                    as_device_type(b.values), b.stride, zero<ValueType>(),
                    as_device_type(c.values), c.stride);
            } else {
                multivector::fill(exec, c, zero<ValueType>());
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           matrix::view::dense<const ValueType> alpha,
           matrix::view::dense<const ValueType> a,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<const ValueType> beta,
           matrix::view::dense<ValueType> c)
{
    using namespace oneapi::mkl;
    if constexpr (onemkl::is_supported<ValueType>::value) {
        if (b.stride != 0 && c.stride != 0) {
            if (a.size[1] > 0 && a.values && b.values && c.values) {
                oneapi::mkl::blas::row_major::gemm(
                    *exec->get_queue(), transpose::nontrans,
                    transpose::nontrans, c.size[0], c.size[1], a.size[1],
                    exec->copy_val_to_host(alpha.values),
                    as_device_type(a.values), a.stride,
                    as_device_type(b.values), b.stride,
                    exec->copy_val_to_host(beta.values),
                    as_device_type(c.values), c.stride);
            } else {
                dense::scale(exec, beta, c);
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec,
                    matrix::view::dense<const ValueType> source,
                    const int64* row_ptrs,
                    matrix::view::coo<ValueType, IndexType> result)
{
    const auto num_rows = result.size[0];
    const auto num_cols = result.size[1];
    const auto in_vals = as_device_type(source.values);
    const auto stride = source.stride;

    auto rows = result.row_idxs;
    auto cols = result.col_idxs;
    auto vals = as_device_type(result.values);

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
                    matrix::view::dense<const ValueType> source,
                    matrix::view::csr<ValueType, IndexType> result)
{
    const auto num_rows = result.size[0];
    const auto num_cols = result.size[1];
    const auto in_vals = as_device_type(source.values);
    const auto stride = source.stride;

    const auto row_ptrs = result.row_ptrs;
    auto cols = result.col_idxs;
    auto vals = as_device_type(result.values);

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
                    matrix::view::dense<const ValueType> source,
                    matrix::view::ell<ValueType, IndexType> result)
{
    const auto num_rows = result.size[0];
    const auto num_cols = result.size[1];
    const auto max_nnz_per_row = result.num_stored_elements_per_row;
    const auto in_vals = as_device_type(source.values);
    const auto in_stride = source.stride;

    auto cols = result.col_idxs;
    auto vals = as_device_type(result.values);
    const auto stride = result.stride;

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
                vals[col_idx * stride + row] = zero<device_type<ValueType>>();
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      matrix::view::dense<const ValueType> source,
                      matrix::Fbcsr<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzero_blocks_per_row(std::shared_ptr<const DefaultExecutor> exec,
                                  matrix::view::dense<const ValueType> source,
                                  int bs,
                                  IndexType* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec,
                       matrix::view::dense<const ValueType> source,
                       const int64* coo_row_ptrs,
                       matrix::Hybrid<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto ell_lim = result->get_ell_num_stored_elements_per_row();
    const auto in_vals = as_device_type(source.values);
    const auto in_stride = source.stride;
    const auto ell_stride = result->get_ell_stride();
    auto ell_cols = result->get_ell_col_idxs();
    auto ell_vals = as_device_type(result->get_ell_values());
    auto coo_rows = result->get_coo_row_idxs();
    auto coo_cols = result->get_coo_col_idxs();
    auto coo_vals = as_device_type(result->get_coo_values());

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
                ell_vals[ell_idx] = zero<device_type<ValueType>>();
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
                      matrix::view::dense<const ValueType> source,
                      matrix::view::sellp<ValueType, IndexType> result)
{
    const auto num_rows = result.size[0];
    const auto num_cols = result.size[1];
    const auto stride = source.stride;
    const auto in_vals = as_device_type(source.values);

    const auto slice_sets = result.slice_sets;
    const auto slice_size = result.slice_size;
    auto vals = as_device_type(result.values);
    auto col_idxs = result.col_idxs;

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
                vals[out_idx] = zero<device_type<ValueType>>();
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const DefaultExecutor> exec,
                             matrix::view::dense<const ValueType> source,
                             matrix::SparsityCsr<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto in_vals = as_device_type(source.values);
    const auto stride = source.stride;

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


}  // namespace dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
