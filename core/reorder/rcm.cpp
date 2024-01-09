// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/rcm.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/reorder/rcm_kernels.hpp"


namespace gko {
namespace reorder {
namespace rcm {
namespace {


GKO_REGISTER_OPERATION(compute_permutation, rcm::compute_permutation);


}  // anonymous namespace
}  // namespace rcm


template <typename ValueType, typename IndexType>
void rcm_reorder(const matrix::SparsityCsr<ValueType, IndexType>* mtx,
                 IndexType* permutation, IndexType* inv_permutation,
                 starting_strategy strategy)
{
    const auto exec = mtx->get_executor();
    const IndexType num_rows = mtx->get_size()[0];
    exec->run(rcm::make_compute_permutation(
        num_rows, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
        permutation, inv_permutation, strategy));
}


template <typename ValueType, typename IndexType>
Rcm<ValueType, IndexType>::Rcm(std::shared_ptr<const Executor> exec)
    : EnablePolymorphicObject<Rcm, ReorderingBase<IndexType>>(std::move(exec))
{}


template <typename ValueType, typename IndexType>
Rcm<ValueType, IndexType>::Rcm(const Factory* factory,
                               const ReorderingBaseArgs& args)
    : EnablePolymorphicObject<Rcm, ReorderingBase<IndexType>>(
          factory->get_executor()),
      parameters_{factory->get_parameters()}
{
    // The reordering is not supported on DPC++, use the host instead
    const auto is_dpcpp_executor = bool(
        std::dynamic_pointer_cast<const DpcppExecutor>(this->get_executor()));
    auto work_exec = is_dpcpp_executor ? this->get_executor()->get_master()
                                       : this->get_executor();

    auto adjacency_matrix = SparsityMatrix::create(work_exec);

    // The adjacency matrix has to be square.
    GKO_ASSERT_IS_SQUARE_MATRIX(args.system_matrix);
    // This is needed because it does not make sense to call the copy and
    // convert if the existing matrix is empty.
    if (args.system_matrix->get_size()) {
        auto tmp =
            copy_and_convert_to<SparsityMatrix>(work_exec, args.system_matrix);
        // This function provided within the Sparsity matrix format removes
        // the diagonal elements and outputs an adjacency matrix.
        adjacency_matrix = tmp->to_adjacency_matrix();
    }

    auto const size = adjacency_matrix->get_size()[0];
    permutation_ = PermutationMatrix::create(work_exec, size);

    // To make it explicit.
    inv_permutation_ = nullptr;
    if (parameters_.construct_inverse_permutation) {
        inv_permutation_ = PermutationMatrix::create(work_exec, size);
    }

    rcm_reorder(
        adjacency_matrix.get(), permutation_->get_permutation(),
        inv_permutation_ ? inv_permutation_->get_permutation() : nullptr,
        parameters_.strategy);

    // Copy back results to gpu if necessary.
    if (is_dpcpp_executor) {
        const auto gpu_exec = this->get_executor();
        auto gpu_perm = share(PermutationMatrix::create(gpu_exec, size));
        gpu_perm->copy_from(permutation_);
        permutation_ = gpu_perm;
        if (inv_permutation_) {
            auto gpu_inv_perm =
                share(PermutationMatrix::create(gpu_exec, size));
            gpu_inv_perm->copy_from(inv_permutation_);
            inv_permutation_ = gpu_inv_perm;
        }
    }
    auto permutation_array =
        make_array_view(this->get_executor(), permutation_->get_size()[0],
                        permutation_->get_permutation());
    this->set_permutation_array(permutation_array);
}


#define GKO_DECLARE_RCM(ValueType, IndexType) class Rcm<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RCM);


}  // namespace reorder


namespace experimental {
namespace reorder {


template <typename IndexType>
Rcm<IndexType>::Rcm(std::shared_ptr<const Executor> exec,
                    const parameters_type& params)
    : EnablePolymorphicObject<Rcm, LinOpFactory>(std::move(exec)),
      parameters_{params}
{}


template <typename IndexType>
std::unique_ptr<matrix::Permutation<IndexType>> Rcm<IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<permutation_type>(static_cast<permutation_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename IndexType>
std::unique_ptr<LinOp> Rcm<IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();
    // The reordering is not supported on DPC++, use the host instead
    const auto is_dpcpp_executor = bool(
        std::dynamic_pointer_cast<const DpcppExecutor>(this->get_executor()));
    auto work_exec = is_dpcpp_executor ? this->get_executor()->get_master()
                                       : this->get_executor();
    const auto num_rows = system_matrix->get_size()[0];
    using sparsity_mtx = matrix::SparsityCsr<float, IndexType>;
    std::unique_ptr<LinOp> converted;
    // extract row pointers and column indices
    const IndexType* row_ptrs{};
    const IndexType* col_idxs{};
    size_type nnz{};
    auto convert = [&](auto op, auto value_type) {
        using ValueType = std::decay_t<decltype(value_type)>;
        using Identity = matrix::Identity<ValueType>;
        using Mtx = matrix::Csr<ValueType, IndexType>;
        using Scalar = matrix::Dense<ValueType>;
        auto conv_csr = Mtx::create(work_exec);
        as<ConvertibleTo<Mtx>>(op)->convert_to(conv_csr);
        if (!parameters_.skip_symmetrize) {
            auto scalar = initialize<Scalar>({one<ValueType>()}, exec);
            auto id = Identity::create(exec, conv_csr->get_size()[0]);
            // compute A^T + A
            conv_csr->transpose()->apply(scalar, id, scalar, conv_csr);
        }
        if (exec != work_exec) {
            conv_csr = gko::clone(work_exec, std::move(conv_csr));
        }
        nnz = conv_csr->get_num_stored_elements();
        row_ptrs = conv_csr->get_const_row_ptrs();
        col_idxs = conv_csr->get_const_col_idxs();
        converted = std::move(conv_csr);
    };
    if (auto convertible =
            dynamic_cast<const ConvertibleTo<matrix::Csr<float, IndexType>>*>(
                system_matrix.get())) {
        convert(system_matrix, float{});
    } else {
        convert(system_matrix, std::complex<float>{});
    }

    array<IndexType> permutation(work_exec, num_rows);

    // remove diagonal entries
    auto pattern = sparsity_mtx::create_const(
        work_exec, gko::dim<2>{num_rows, num_rows},
        make_const_array_view(work_exec, nnz, col_idxs),
        make_const_array_view(work_exec, num_rows + 1, row_ptrs));
    pattern = pattern->to_adjacency_matrix();
    rcm_reorder(pattern.get(), permutation.get_data(),
                static_cast<IndexType*>(nullptr), parameters_.strategy);

    // permutation gets copied to device via gko::array constructor
    return permutation_type::create(exec, std::move(permutation));
}


#undef GKO_DECLARE_RCM
#define GKO_DECLARE_RCM(IndexType) class Rcm<IndexType>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM);


}  // namespace reorder
}  // namespace experimental
}  // namespace gko
