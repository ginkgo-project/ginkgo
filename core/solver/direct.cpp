// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/direct.hpp>


#include <memory>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/factorization/factorization.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace experimental {
namespace solver {


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Direct<ValueType, IndexType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Direct<ValueType, IndexType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
Direct<ValueType, IndexType>::Direct(const Direct& other)
    : EnableLinOp<Direct>{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename IndexType>
Direct<ValueType, IndexType>::Direct(Direct&& other)
    : EnableLinOp<Direct>{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
Direct<ValueType, IndexType>& Direct<ValueType, IndexType>::operator=(
    const Direct& other)
{
    if (this != &other) {
        EnableLinOp<Direct>::operator=(other);
        gko::solver::EnableSolverBase<Direct, factorization_type>::operator=(
            other);
        const auto exec = this->get_executor();
        lower_solver_ = other.lower_solver_->clone(exec);
        upper_solver_ = other.upper_solver_->clone(exec);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Direct<ValueType, IndexType>& Direct<ValueType, IndexType>::operator=(
    Direct&& other)
{
    if (this != &other) {
        EnableLinOp<Direct>::operator=(std::move(other));
        gko::solver::EnableSolverBase<Direct, factorization_type>::operator=(
            std::move(other));
        const auto exec = this->get_executor();
        lower_solver_ = std::move(other.lower_solver_);
        upper_solver_ = std::move(other.upper_solver_);
    }
    return *this;
}


template <typename ValueType, typename IndexType>
Direct<ValueType, IndexType>::Direct(std::shared_ptr<const Executor> exec)
    : EnableLinOp<Direct>{exec}
{}


template <typename ValueType, typename IndexType>
static std::shared_ptr<const factorization::Factorization<ValueType, IndexType>>
generate_factorization(
    std::shared_ptr<const LinOpFactory> factorization_factory,
    std::shared_ptr<const LinOp> system_matrix)
{
    if (auto factorization = std::dynamic_pointer_cast<
            const factorization::Factorization<ValueType, IndexType>>(
            system_matrix)) {
        return factorization;
    } else {
        return as<factorization::Factorization<ValueType, IndexType>>(
            factorization_factory->generate(system_matrix));
    }
}


template <typename ValueType, typename IndexType>
Direct<ValueType, IndexType>::Direct(const Factory* factory,
                                     std::shared_ptr<const LinOp> system_matrix)
    : EnableLinOp<Direct>{factory->get_executor(), system_matrix->get_size()},
      gko::solver::EnableSolverBase<
          Direct, factorization::Factorization<ValueType, IndexType>>{
          generate_factorization<ValueType, IndexType>(
              factory->get_parameters().factorization, system_matrix)}
{
    using factorization::storage_type;
    const auto factors = this->get_system_matrix();
    const auto exec = this->get_executor();
    const auto type = factors->get_storage_type();
    const bool lower_unit_diag = type == storage_type::combined_lu ||
                                 type == storage_type::combined_ldu ||
                                 type == storage_type::symm_combined_ldl;
    const bool upper_unit_diag = type == storage_type::combined_ldu ||
                                 type == storage_type::symm_combined_ldl;
    const bool separate_diag = factors->get_diagonal() ||
                               type == storage_type::combined_ldu ||
                               type == storage_type::symm_combined_ldl;
    if (separate_diag) {
        GKO_NOT_SUPPORTED(type);
    }
    const auto num_rhs = factory->get_parameters().num_rhs;
    const auto lower_factory = lower_type::build()
                                   .with_num_rhs(num_rhs)
                                   .with_unit_diagonal(lower_unit_diag)
                                   .on(exec);
    const auto upper_factory = upper_type::build()
                                   .with_num_rhs(num_rhs)
                                   .with_unit_diagonal(upper_unit_diag)
                                   .on(exec);
    switch (type) {
    case storage_type::empty:
        // remove the factor storage entirely
        this->clear();
        break;
    case storage_type::composition:
    case storage_type::symm_composition:
        // TODO handle diagonal
        lower_solver_ = lower_factory->generate(factors->get_lower_factor());
        upper_solver_ = upper_factory->generate(factors->get_upper_factor());
        break;
    case storage_type::combined_lu:
    case storage_type::combined_ldu:
    case storage_type::symm_combined_cholesky:
    case storage_type::symm_combined_ldl:
        lower_solver_ = lower_factory->generate(factors->get_combined());
        upper_solver_ = upper_factory->generate(factors->get_combined());
        break;
    }
}


template <typename ValueType, typename IndexType>
void Direct<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix() || !this->lower_solver_ ||
        !this->upper_solver_) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            using Vector = matrix::Dense<ValueType>;
            using ws = gko::solver::workspace_traits<Direct>;
            this->setup_workspace();
            auto intermediate = this->create_workspace_op_with_config_of(
                ws::intermediate, dense_b);
            lower_solver_->apply(dense_b, intermediate);
            upper_solver_->apply(intermediate, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Direct<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                              const LinOp* b, const LinOp* beta,
                                              LinOp* x) const
{
    if (!this->get_system_matrix() || !this->lower_solver_ ||
        !this->upper_solver_) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            using Vector = matrix::Dense<ValueType>;
            using ws = gko::solver::workspace_traits<Direct>;
            this->setup_workspace();
            auto intermediate = this->create_workspace_op_with_config_of(
                ws::intermediate, dense_b);
            lower_solver_->apply(dense_b, intermediate);
            upper_solver_->apply(dense_alpha, intermediate, dense_beta,
                                 dense_x);
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_DIRECT(ValueType, IndexType) \
    class Direct<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DIRECT);


}  // namespace solver
}  // namespace experimental


namespace solver {


template <typename ValueType, typename IndexType>
int workspace_traits<gko::experimental::solver::Direct<ValueType, IndexType>>::
    num_arrays(const Solver&)
{
    return 0;
}


template <typename ValueType, typename IndexType>
int workspace_traits<gko::experimental::solver::Direct<ValueType, IndexType>>::
    num_vectors(const Solver&)
{
    return 1;
}


template <typename ValueType, typename IndexType>
std::vector<std::string> workspace_traits<gko::experimental::solver::Direct<
    ValueType, IndexType>>::op_names(const Solver&)
{
    return {"intermediate"};
}


template <typename ValueType, typename IndexType>
std::vector<std::string> workspace_traits<gko::experimental::solver::Direct<
    ValueType, IndexType>>::array_names(const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<gko::experimental::solver::Direct<
    ValueType, IndexType>>::scalars(const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<gko::experimental::solver::Direct<
    ValueType, IndexType>>::vectors(const Solver&)
{
    return {intermediate};
}


#define GKO_DECLARE_DIRECT_TRAITS(ValueType, IndexType) \
    class workspace_traits<                             \
        gko::experimental::solver::Direct<ValueType, IndexType>>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DIRECT_TRAITS);


}  // namespace solver
}  // namespace gko
