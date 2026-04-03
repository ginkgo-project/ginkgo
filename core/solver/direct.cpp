// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/direct.hpp"

#include <memory>
#include <string>

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/factorization/factorization.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

#include "core/config/config_helper.hpp"
#include "core/solver/direct_kernels.hpp"


namespace gko {
namespace experimental {
namespace solver {


#if GKO_HAVE_CUDSS
namespace direct_dispatch {
namespace {


GKO_REGISTER_OPERATION(vendor_generate, direct::generate);
GKO_REGISTER_OPERATION(vendor_solve, direct::solve);


}  // anonymous namespace
}  // namespace direct_dispatch
#endif  // GKO_HAVE_CUDSS


template <typename ValueType, typename IndexType>
typename Direct<ValueType, IndexType>::parameters_type
Direct<ValueType, IndexType>::parse(const config::pnode& config,
                                    const config::registry& context,
                                    const config::type_descriptor& td_for_child)
{
    auto params = Direct<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("num_rhs")) {
        params.with_num_rhs(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = config_check.get("factorization")) {
        params.with_factorization(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config_check.get("algorithm")) {
        auto alg_str = obj.get_string();
        if (alg_str == "factorization") {
            params.with_algorithm(direct_algorithm::factorization);
        } else if (alg_str == "vendor") {
            params.with_algorithm(direct_algorithm::vendor);
        } else {
            GKO_INVALID_CONFIG_VALUE("algorithm", alg_str);
        }
    }
    if (auto& obj = config_check.get("vendor_params")) {
        vendor_parameters vp{};
        config::config_check_decorator vp_check(obj);
        if (auto& mt = vp_check.get("matrix_type")) {
            vp.matrix_type = gko::config::get_value<int>(mt);
        }
        if (auto& mv = vp_check.get("matrix_view")) {
            vp.matrix_view = gko::config::get_value<int>(mv);
        }
        if (auto& ra = vp_check.get("reordering_alg")) {
            vp.reordering_alg = gko::config::get_value<int>(ra);
        }
        if (auto& he = vp_check.get("hybrid_execute")) {
            vp.hybrid_execute = gko::config::get_value<bool>(he);
        }
        if (auto& hm = vp_check.get("hybrid_memory")) {
            vp.hybrid_memory = gko::config::get_value<bool>(hm);
        }
        params.with_vendor_params(vp);
    }

    return params;
}


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
        vendor_state_ = other.vendor_state_;
        vendor_system_matrix_ = other.vendor_system_matrix_;
        if (other.lower_solver_) {
            const auto exec = this->get_executor();
            lower_solver_ = other.lower_solver_->clone(exec);
            upper_solver_ = other.upper_solver_->clone(exec);
        } else {
            lower_solver_ = nullptr;
            upper_solver_ = nullptr;
        }
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
        vendor_state_ = std::move(other.vendor_state_);
        vendor_system_matrix_ = std::move(other.vendor_system_matrix_);
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
      gko::solver::EnableSolverBase<Direct, factorization_type>{}
{
    using factorization::storage_type;
    const auto exec = this->get_executor();
    const auto& params = factory->get_parameters();

#if GKO_HAVE_CUDSS
    if (params.algorithm == direct_algorithm::vendor) {
        using CsrType = matrix::Csr<ValueType, IndexType>;
        auto csr = copy_and_convert_to<CsrType>(exec, system_matrix);
        // Keep CSR alive — cuDSS references its data via zero-copy
        vendor_system_matrix_ = csr;

        exec->run(direct_dispatch::make_vendor_generate(
            csr.get(), vendor_state_, params.vendor_params));
        return;
    }
#endif  // GKO_HAVE_CUDSS

    // Default: Factorization path
    auto factors = generate_factorization<ValueType, IndexType>(
        params.factorization, system_matrix);
    this->set_system_matrix(factors);

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
    const auto num_rhs = params.num_rhs;
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
#if GKO_HAVE_CUDSS
    if (vendor_state_) {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_b, auto dense_x) {
                using Dense = matrix::Dense<ValueType>;
                const auto exec = this->get_executor();
                const auto nrhs = dense_b->get_size()[1];
                if (nrhs <= 1) {
                    exec->run(direct_dispatch::make_vendor_solve(
                        vendor_state_.get(), dense_b, dense_x));
                } else {
                    const auto nrows = dense_b->get_size()[0];
                    auto tmp_b = Dense::create(exec, dim<2>{nrows, 1});
                    auto tmp_x = Dense::create(exec, dim<2>{nrows, 1});
                    auto mut_b = const_cast<std::remove_const_t<
                        std::remove_pointer_t<decltype(dense_b)>>*>(dense_b);
                    for (size_type j = 0; j < nrhs; ++j) {
                        mut_b->create_submatrix(span{0, nrows}, span{j, j + 1})
                            ->convert_to(tmp_b);
                        exec->run(direct_dispatch::make_vendor_solve(
                            vendor_state_.get(), tmp_b.get(), tmp_x.get()));
                        dense_x
                            ->create_submatrix(span{0, nrows}, span{j, j + 1})
                            ->copy_from(tmp_x);
                    }
                }
            },
            b, x);
        return;
    }
#endif  // GKO_HAVE_CUDSS

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
#if GKO_HAVE_CUDSS
    if (vendor_state_) {
        precision_dispatch_real_complex<ValueType>(
            [this](auto dense_alpha, auto dense_b, auto dense_beta,
                   auto dense_x) {
                auto tmp = dense_x->clone();
                this->apply_impl(dense_b, tmp.get());
                dense_x->scale(dense_beta);
                dense_x->add_scaled(dense_alpha, tmp);
            },
            alpha, b, beta, x);
        return;
    }
#endif  // GKO_HAVE_CUDSS

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
    struct workspace_traits<                            \
        gko::experimental::solver::Direct<ValueType, IndexType>>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DIRECT_TRAITS);


}  // namespace solver
}  // namespace gko
