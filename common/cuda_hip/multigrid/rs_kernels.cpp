// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// #include "core/multigrid/rs_kernels.hpp"

// #include <memory>

// #include <thrust/device_ptr.h>
// #include <thrust/iterator/zip_iterator.h>
// #include <thrust/reduce.h>
// #include <thrust/sort.h>
// #include <thrust/tuple.h>

// #include <ginkgo/core/base/exception_helpers.hpp>
// #include <ginkgo/core/base/math.hpp>

// #include "common/cuda_hip/base/thrust.hpp"
// #include "common/cuda_hip/base/types.hpp"
// #include "common/cuda_hip/components/memory.hpp"
// #include "common/cuda_hip/components/thread_ids.hpp"


// namespace gko {
// namespace kernels {
// namespace GKO_DEVICE_NAMESPACE {
// /**
//  * @brief The RS solver namespace.
//  *
//  * @ingroup rs
//  */
// namespace rs {

// template <typename ValueType, typename IndexType>
// void compute_soc_row_ptrs(
//     std::shared_ptr<const DefaultExecutor> exec,
//     const matrix::Csr<ValueType, IndexType>* A,
//     remove_complex<ValueType> theta,
//     IndexType* row_ptrs)
// {
//     GKO_NOT_IMPLEMENTED;
// }

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_RS_COMPUTE_SOC_ROW_PTRS_KERNEL);


// template <typename ValueType, typename IndexType>
// void fill_soc(
//     std::shared_ptr<const DefaultExecutor> exec,
//     const matrix::Csr<ValueType, IndexType>* A,
//     remove_complex<ValueType> theta,
//     matrix::Csr<ValueType, IndexType>* S)
// {
//     GKO_NOT_IMPLEMENTED;
// }

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_RS_FILL_SOC_KERNEL);


// // Compute lambda_i = number of strong neighbors
// template <typename ValueType, typename IndexType>
// void compute_lambda(std::shared_ptr<const DefaultExecutor> exec,
//                     const matrix::Csr<ValueType, IndexType>* S,
//                     IndexType* lambda)
// {
//     GKO_NOT_IMPLEMENTED;
// }

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
//     GKO_DECLARE_RS_COMPUTE_LAMBDA_KERNEL);


// // Init all nodes as undecided (0)
// template <typename IndexType>
// void init_cf(std::shared_ptr<const DefaultExecutor> exec,
//              array<IndexType>& cf_marker)
// {
//     GKO_NOT_IMPLEMENTED;
// }

// GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RS_INIT_CF_KERNEL);


// // Classical RS greedy selection
// template <typename ValueType, typename IndexType>
// void rs_coarsening(std::shared_ptr<const DefaultExecutor> exec,
//                    const matrix::Csr<ValueType, IndexType>* S,
//                    IndexType* lambda, array<IndexType>& cf_marker)
// {
//     GKO_NOT_IMPLEMENTED;
// }

// GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RS_COARSENING_KERNEL);


// // Cleanup: ensure no undecided remain (make them F)
// template <typename IndexType>
// void rs_cleanup(std::shared_ptr<const DefaultExecutor> exec,
//                 array<IndexType>& cf_marker)
// {
//     GKO_NOT_IMPLEMENTED;
// }

// GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RS_CLEANUP_KERNEL);


// // Count C-points
// template <typename IndexType>
// void count_coarse(std::shared_ptr<const DefaultExecutor> exec,
//                   const array<IndexType>& cf_marker, IndexType* coarse_size)
// {
//     GKO_NOT_IMPLEMENTED;
// }

// GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RS_COUNT_COARSE_KERNEL);


// // Fill coarse row index array
// template <typename IndexType>
// void fill_coarse_rows(std::shared_ptr<const DefaultExecutor> exec,
//                       const array<IndexType>& cf_marker, IndexType*
//                       coarse_rows)
// {
//     GKO_NOT_IMPLEMENTED;
// }

// GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RS_FILL_COARSE_ROWS_KERNEL);

// }  // namespace rs
// }  // namespace GKO_DEVICE_NAMESPACE
// }  // namespace kernels
// }  // namespace gko
