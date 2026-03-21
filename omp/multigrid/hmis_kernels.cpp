// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/hmis_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {
namespace omp {
namespace hmis {


template <typename ValueType, typename IndexType>
void compute_strength(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* system_matrix,
                      remove_complex<ValueType> theta,
                      IndexType* strength_row_ptrs,
                      array<IndexType>& strength_col_idxs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HMIS_COMPUTE_STRENGTH_KERNEL);


template <typename ValueType, typename IndexType>
void ruge_stueben_first_pass(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* strength,
    const matrix::Csr<ValueType, IndexType>* strength_transpose,
    array<IndexType>& cf_marker) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HMIS_RUGE_STUEBEN_FIRST_PASS_KERNEL);


template <typename ValueType, typename IndexType>
void pmis_coarsen(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* strength,
                  const matrix::Csr<ValueType, IndexType>* strength_transpose,
                  unsigned max_iterations,
                  array<IndexType>& cf_marker) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HMIS_PMIS_COARSEN_KERNEL);


template <typename IndexType>
void build_cpoint_map(std::shared_ptr<const DefaultExecutor> exec,
                      const array<IndexType>& cf_marker,
                      IndexType* num_c_points,
                      array<IndexType>& c_point_map) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_HMIS_BUILD_CPOINT_MAP_KERNEL);


template <typename ValueType, typename IndexType>
void build_prolongation(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    const matrix::Csr<ValueType, IndexType>* strength,
    const array<IndexType>& cf_marker, IndexType num_c_points,
    const array<IndexType>& c_point_map, IndexType* prolongation_row_ptrs,
    array<IndexType>& prolongation_col_idxs,
    array<ValueType>& prolongation_vals) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HMIS_BUILD_PROLONGATION_KERNEL);


}  // namespace hmis
}  // namespace omp
}  // namespace kernels
}  // namespace gko
