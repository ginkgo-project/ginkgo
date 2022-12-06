/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
    : EnableLinOp<Direct>{other},
      gko::solver::EnableSolverBase<Direct, factorization_type>{other},
      lower_solver_{other.lower_solver_->clone()},
      upper_solver_{other.upper_solver_->clone()}
{}


template <typename ValueType, typename IndexType>
Direct<ValueType, IndexType>::Direct(Direct&& other)
    : EnableLinOp<Direct>{std::move(other)},
      gko::solver::EnableSolverBase<Direct, factorization_type>{
          std::move(other)},
      lower_solver_{std::move(other.lower_solver_)},
      upper_solver_{std::move(other.upper_solver_)}
{}


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
    const auto lower_factory = lower_type::build().on(factory->get_executor());
    const auto upper_factory = upper_type::build().on(factory->get_executor());
    const auto factors = this->get_system_matrix();
    switch (factors->get_storage_type()) {
    case storage_type::empty:
        // leave both solvers as nullptr
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
        // TODO handle diagonal
        lower_solver_ = lower_factory->generate(factors->get_combined());
        upper_solver_ = upper_factory->generate(factors->get_combined());
        break;
    }
}


template <typename ValueType, typename IndexType>
void Direct<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            using Vector = matrix::Dense<ValueType>;
            using ws = gko::solver::workspace_traits<Direct>;
            const auto exec = this->get_executor();
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
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            using Vector = matrix::Dense<ValueType>;
            using ws = gko::solver::workspace_traits<Direct>;
            const auto exec = this->get_executor();
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
