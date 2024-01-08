// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/amd.hpp>


#include <cstddef>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace experimental {
namespace reorder {
namespace suitesparse_wrapper {


using std::int32_t;
using std::int64_t;
using std::size_t;


#include "third_party/SuiteSparse/AMD/Include/amd.h"


void amd_reorder(int32 num_rows, int32* row_ptrs,
                 int32* col_idxs_plus_workspace, int32* row_lengths,
                 int32 workspace_size, int32* nv, int32* next, int32* last,
                 int32* head, int32* elen, int32* degree, int32* w)
{
    std::array<double, AMD_CONTROL> control{};
    std::array<double, AMD_INFO> info{};
    amd_defaults(control.data());
    amd_2(num_rows, row_ptrs, col_idxs_plus_workspace, row_lengths,
          workspace_size, row_ptrs[num_rows], nv, next, last, head, elen,
          degree, w, control.data(), info.data());
}


void amd_reorder(int64 num_rows, int64* row_ptrs,
                 int64* col_idxs_plus_workspace, int64* row_lengths,
                 int64 workspace_size, int64* nv, int64* next, int64* last,
                 int64* head, int64* elen, int64* degree, int64* w)
{
    std::array<double, AMD_CONTROL> control{};
    std::array<double, AMD_INFO> info{};
    amd_l_defaults(control.data());
    amd_l2(num_rows, row_ptrs, col_idxs_plus_workspace, row_lengths,
           workspace_size, row_ptrs[num_rows], nv, next, last, head, elen,
           degree, w, control.data(), info.data());
}


GKO_REGISTER_HOST_OPERATION(amd_reorder, amd_reorder);


}  // namespace suitesparse_wrapper


template <typename IndexType>
Amd<IndexType>::Amd(std::shared_ptr<const Executor> exec,
                    const parameters_type& params)
    : EnablePolymorphicObject<Amd, LinOpFactory>(std::move(exec)),
      parameters_{params}
{}


template <typename IndexType>
std::unique_ptr<matrix::Permutation<IndexType>> Amd<IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<permutation_type>(static_cast<permutation_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename IndexType>
std::unique_ptr<LinOp> Amd<IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();
    const auto num_rows = system_matrix->get_size()[0];
    using complex_scalar = matrix::Dense<std::complex<float>>;
    using real_scalar = matrix::Dense<float>;
    using complex_identity = matrix::Identity<std::complex<float>>;
    using real_identity = matrix::Identity<float>;
    using complex_mtx = matrix::Csr<std::complex<float>, IndexType>;
    using real_mtx = matrix::Csr<float, IndexType>;
    using sparsity_mtx = matrix::SparsityCsr<float, IndexType>;
    std::unique_ptr<LinOp> converted;
    // extract row pointers and column indices
    IndexType* d_row_ptrs{};
    IndexType* d_col_idxs{};
    size_type d_nnz{};
    if (auto convertible = dynamic_cast<const ConvertibleTo<complex_mtx>*>(
            system_matrix.get())) {
        auto conv_csr = complex_mtx::create(exec);
        convertible->convert_to(conv_csr);
        if (!parameters_.skip_sorting) {
            conv_csr->sort_by_column_index();
        }
        if (!parameters_.skip_symmetrize) {
            auto scalar =
                initialize<complex_scalar>({one<std::complex<float>>()}, exec);
            auto id = complex_identity::create(exec, conv_csr->get_size()[0]);
            // compute A^T + A
            conv_csr->transpose()->apply(scalar, id, scalar, conv_csr);
        }
        d_nnz = conv_csr->get_num_stored_elements();
        d_row_ptrs = conv_csr->get_row_ptrs();
        d_col_idxs = conv_csr->get_col_idxs();
        converted = std::move(conv_csr);
    } else {
        auto conv_csr = real_mtx::create(exec);
        as<ConvertibleTo<real_mtx>>(system_matrix)->convert_to(conv_csr);
        if (!parameters_.skip_sorting) {
            conv_csr->sort_by_column_index();
        }
        if (!parameters_.skip_symmetrize) {
            auto scalar = initialize<real_scalar>({one<float>()}, exec);
            auto id = real_identity::create(exec, conv_csr->get_size()[0]);
            // compute A^T + A
            conv_csr->transpose()->apply(scalar, id, scalar, conv_csr);
        }
        d_nnz = conv_csr->get_num_stored_elements();
        d_row_ptrs = conv_csr->get_row_ptrs();
        d_col_idxs = conv_csr->get_col_idxs();
        converted = std::move(conv_csr);
    }

    // remove diagonal entries
    auto pattern =
        sparsity_mtx::create(exec, gko::dim<2>{num_rows, num_rows},
                             make_array_view(exec, d_nnz, d_col_idxs),
                             make_array_view(exec, num_rows + 1, d_row_ptrs));
    pattern = pattern->to_adjacency_matrix();
    // copy data to the CPU
    array<IndexType> row_ptrs{host_exec, num_rows + 1};
    host_exec->copy_from(exec, num_rows + 1, pattern->get_const_row_ptrs(),
                         row_ptrs.get_data());
    const auto nnz = row_ptrs.get_data()[num_rows];
    // we use this much space for the column index workspace, the rest for
    // row workspace
    const auto col_idxs_plus_workspace_size = nnz + nnz / 5 + 2 * num_rows;
    array<IndexType> col_idxs_plus_workspace{
        host_exec, col_idxs_plus_workspace_size + 6 * num_rows};
    host_exec->copy_from(exec, nnz, pattern->get_const_col_idxs(),
                         col_idxs_plus_workspace.get_data());

    array<IndexType> permutation{host_exec, num_rows};
    array<IndexType> row_lengths{host_exec, num_rows};
    for (size_type row = 0; row < num_rows; row++) {
        row_lengths.get_data()[row] =
            row_ptrs.get_data()[row + 1] - row_ptrs.get_data()[row];
    }
    // different temporary workspace arrays for amd
    const auto last = permutation.get_data();
    const auto nv =
        col_idxs_plus_workspace.get_data() + col_idxs_plus_workspace_size;
    const auto next = nv + num_rows;
    const auto head = next + num_rows;
    const auto elen = head + num_rows;
    const auto degree = elen + num_rows;
    const auto w = degree + num_rows;
    // run AMD
    exec->run(suitesparse_wrapper::make_amd_reorder(
        static_cast<IndexType>(num_rows), row_ptrs.get_data(),
        col_idxs_plus_workspace.get_data(), row_lengths.get_data(),
        static_cast<IndexType>(col_idxs_plus_workspace_size), nv, next, last,
        head, elen, degree, w));

    // permutation gets copied to device via gko::array constructor
    return permutation_type::create(exec, std::move(permutation));
}


#define GKO_DECLARE_AMD(IndexType) class Amd<IndexType>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMD);


}  // namespace reorder
}  // namespace experimental
}  // namespace gko
