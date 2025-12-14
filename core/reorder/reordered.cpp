// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/reorder/reordered.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace experimental {
namespace reorder {


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOpFactory::ReuseData>
Reordered<ValueType, IndexType>::Factory::create_empty_reuse_data() const
{
    return std::make_unique<ReorderedReuseData>();
}


template <typename ValueType, typename IndexType>
auto Reordered<ValueType, IndexType>::Factory::generate_reuse(
    std::shared_ptr<const LinOp> input, BaseReuseData& reuse_data) const
    -> std::unique_ptr<Reordered>
{
    return as<Reordered>(LinOpFactory::generate_reuse(input, reuse_data));
}


template <typename ValueType, typename IndexType>
bool Reordered<ValueType, IndexType>::Factory::ReorderedReuseData::is_empty()
    const
{
    // here we could also assert that all the other members are empty,
    // but since we are the only class accessing this data, we should be fine.
    return !this->permutation_;
}


template <typename ValueType, typename IndexType>
void Reordered<ValueType, IndexType>::Factory::check_reuse_consistent(
    const LinOp* input, BaseReuseData& reuse_data) const
{
    auto& rrd = *as<ReorderedReuseData>(&reuse_data);
    if (rrd.is_empty()) {
        return;
    }
    auto exec = this->get_executor();
    auto csr_input = as<matrix_type>(input);
    auto nnz = csr_input->get_num_stored_elements();

    GKO_ASSERT_IS_SQUARE_MATRIX(input);
    if (rrd.permutation_->get_executor() != exec) {
        throw NotSupported{__FILE__, __LINE__, __func__,
                           name_demangling::get_dynamic_type(*exec)};
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(rrd.permutation_, input);
    GKO_ASSERT_EQUAL_DIMENSIONS(rrd.permuted_, input);
    GKO_ASSERT_EQ(rrd.permuted_->get_num_stored_elements(), nnz);
    GKO_ASSERT_EQ(rrd.permute_reuse_.value_permutation->get_size()[0], nnz);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp>
Reordered<ValueType, IndexType>::Factory::generate_reuse_impl(
    std::shared_ptr<const LinOp> input, BaseReuseData& reuse_data) const
{
    return std::unique_ptr<Reordered<ValueType, IndexType>>(new Reordered{
        this, input, static_cast<ReorderedReuseData&>(reuse_data)});
}


template <typename ValueType, typename IndexType>
void Reordered<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    using matrix::permute_mode;
    if (!this->get_size()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            auto exec = this->get_executor();
            if (this->cache_in_.vec == nullptr ||
                this->cache_in_.vec->get_size() != dense_b->get_size()) {
                // since the operator is square, we can only differ in the
                // number of rhs, so dense_x and dense_b and the internal caches
                // have the same size.
                this->cache_in_.init(exec, dense_b->get_size());
                this->cache_out_.init(exec, dense_b->get_size());
            }
            // A_inner = P A P^T
            // => A b = P^T A_inner P b
            if (inner_op_->apply_uses_initial_guess()) {
                dense_x->permute(this->permutation_, this->cache_out_.get(),
                                 permute_mode::rows);
            }
            dense_b->permute(this->permutation_, this->cache_in_.get(),
                             permute_mode::rows);
            inner_op_->apply(this->cache_in_.get(), this->cache_out_.get());
            this->cache_out_->permute(this->permutation_, dense_x,
                                      permute_mode::inverse_rows);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Reordered<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                 const LinOp* b,
                                                 const LinOp* beta,
                                                 LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
auto Reordered<ValueType, IndexType>::get_parameters() const
    -> const parameters_type&
{
    return parameters_;
}


template <typename ValueType, typename IndexType>
auto Reordered<ValueType, IndexType>::build() -> parameters_type
{
    return {};
}


template <typename ValueType, typename IndexType>
auto Reordered<ValueType, IndexType>::get_permutation() const
    -> std::shared_ptr<const permutation_type>
{
    return permutation_;
}


template <typename ValueType, typename IndexType>
std::shared_ptr<const LinOp>
Reordered<ValueType, IndexType>::get_inner_operator() const
{
    return inner_op_;
}


template <typename ValueType, typename IndexType>
Reordered<ValueType, IndexType>::Reordered(std::shared_ptr<const Executor> exec)
    : EnableLinOp<Reordered>{exec}
{}


template <typename ValueType, typename IndexType>
Reordered<ValueType, IndexType>::Reordered(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix)
    : EnableLinOp<Reordered>{factory->get_executor(),
                             system_matrix->get_size()},
      parameters_{factory->get_parameters()}
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    permutation_ = as<permutation_type>(
        this->get_parameters().reordering->generate(system_matrix));
    auto permuted =
        share(as<matrix_type>(system_matrix)->permute(permutation_));
    inner_op_ = this->get_parameters().inner_operator->generate(permuted);
}


template <typename ValueType, typename IndexType>
Reordered<ValueType, IndexType>::Reordered(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix,
    typename Factory::ReorderedReuseData& rrd)
    : EnableLinOp<Reordered>{factory->get_executor(),
                             system_matrix->get_size()},
      parameters_{factory->get_parameters()}
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    // here system_matrix and this have the same executor, so we don't need to
    // rely on the matrix's executor for array views.
    auto exec = this->get_executor();
    auto csr_matrix = as<matrix_type>(system_matrix);
    auto num_rows = this->get_size()[0];
    auto nnz = csr_matrix->get_num_stored_elements();
    auto from_scratch = rrd.is_empty();
    if (from_scratch) {
        rrd.permutation_ = as<permutation_type>(
            this->get_parameters().reordering->generate(system_matrix));
        std::tie(rrd.permuted_, rrd.permute_reuse_) =
            csr_matrix->permute_reuse(rrd.permutation_);
    }
    permutation_ = rrd.permutation_;
    auto permuted_value_array = array<value_type>(exec, nnz);
    // we need to start off with a mutable matrix to allow using the
    // permute_reuse functionality
    auto permuted = share(matrix_type::create(
        exec, system_matrix->get_size(), std::move(permuted_value_array),
        detail::array_const_cast(make_const_array_view(
            exec, nnz, rrd.permuted_->get_const_col_idxs())),
        detail::array_const_cast(make_const_array_view(
            exec, num_rows + 1, rrd.permuted_->get_const_row_ptrs()))));
    rrd.permute_reuse_.update_values(csr_matrix, permuted);
    // now we have the matrix, we can update the inner operator
    if (from_scratch) {
        // this branch creates a mutable object
        auto inner_reuse =
            this->get_parameters().inner_operator->create_empty_reuse_data();
        inner_op_ = this->get_parameters().inner_operator->generate_reuse(
            permuted, *inner_reuse);
        rrd.inner_reuse_ = std::move(inner_reuse);
    } else {
        // this branch uses an existing constant object
        inner_op_ = this->get_parameters().inner_operator->generate_reuse(
            permuted, *rrd.inner_reuse_);
    }
}


#define GKO_DECLARE_REORDERED(ValueType, IndexType) \
    class Reordered<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_REORDERED);


}  // namespace reorder
}  // namespace experimental
}  // namespace gko
