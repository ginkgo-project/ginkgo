// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/preconditioner/sor.hpp"

#include <set>
#include <string>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/base/array_access.hpp"
#include "core/base/utils.hpp"
#include "core/config/config_helper.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/preconditioner/sor_kernels.hpp"

namespace gko {
namespace preconditioner {
namespace {


GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_weighted_l, sor::initialize_weighted_l);
GKO_REGISTER_OPERATION(initialize_weighted_l_u, sor::initialize_weighted_l_u);


}  // namespace


template <typename ValueType, typename IndexType>
typename Sor<ValueType, IndexType>::parameters_type
Sor<ValueType, IndexType>::parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child)
{
    std::set<std::string> allowed_keys{"skip_sorting", "symmetric",
                                       "relaxation_factor", "l_solver",
                                       "u_solver"};
    gko::config::check_allowed_keys(config, allowed_keys);

    auto params = Sor::build();

    if (auto& obj = config.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("symmetric")) {
        params.with_symmetric(config::get_value<bool>(obj));
    }
    if (auto& obj = config.get("relaxation_factor")) {
        params.with_relaxation_factor(
            config::get_value<remove_complex<ValueType>>(obj));
    }
    if (auto& obj = config.get("l_solver")) {
        params.with_l_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }
    if (auto& obj = config.get("u_solver")) {
        params.with_u_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }

    return params;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Sor<ValueType, IndexType>::composition_type>
Sor<ValueType, IndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix) const
{
    auto product =
        std::unique_ptr<composition_type>(static_cast<composition_type*>(
            this->LinOpFactory::generate(std::move(system_matrix)).release()));
    return product;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> Sor<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system_matrix) const
{
    using Csr = matrix::Csr<ValueType, IndexType>;
    using LTrs = solver::LowerTrs<value_type, index_type>;
    using UTrs = solver::UpperTrs<value_type, index_type>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    auto exec = this->get_executor();
    auto size = system_matrix->get_size();

    auto csr_matrix = convert_to_with_sorting<Csr>(exec, system_matrix,
                                                   parameters_.skip_sorting);

    auto l_trs_factory =
        parameters_.l_solver ? parameters_.l_solver : LTrs::build().on(exec);

    if (parameters_.symmetric) {
        auto u_trs_factory = parameters_.u_solver ? parameters_.u_solver
                                                  : UTrs::build().on(exec);

        array<index_type> l_row_ptrs{exec, size[0] + 1};
        array<index_type> u_row_ptrs{exec, size[0] + 1};
        exec->run(make_initialize_row_ptrs_l_u(
            csr_matrix.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));
        const auto l_nnz =
            static_cast<size_type>(get_element(l_row_ptrs, size[0]));
        const auto u_nnz =
            static_cast<size_type>(get_element(u_row_ptrs, size[0]));

        // create matrices
        auto l_mtx =
            Csr::create(exec, size, array<value_type>{exec, l_nnz},
                        array<index_type>{exec, l_nnz}, std::move(l_row_ptrs));
        auto u_mtx =
            Csr::create(exec, size, array<value_type>{exec, u_nnz},
                        array<index_type>{exec, u_nnz}, std::move(u_row_ptrs));

        // fill l_mtx with 1/w (D + wL)
        // fill u_mtx with 1/(1-w) (D + wU)
        exec->run(make_initialize_weighted_l_u(csr_matrix.get(),
                                               parameters_.relaxation_factor,
                                               l_mtx.get(), u_mtx.get()));

        // scale u_mtx with D^-1
        auto diag = csr_matrix->extract_diagonal();
        diag->inverse_apply(u_mtx, u_mtx);

        // invert the triangular matrices with triangular solvers
        auto l_trs = l_trs_factory->generate(std::move(l_mtx));
        auto u_trs = u_trs_factory->generate(std::move(u_mtx));

        // return (1/(w * (1 - w)) (D + wL) D^-1 (D + wU))^-1
        // because of the inversion, the factor order is switched
        return composition_type::create(std::move(u_trs), std::move(l_trs));
    } else {
        array<index_type> l_row_ptrs{exec, size[0] + 1};
        exec->run(make_initialize_row_ptrs_l(csr_matrix.get(),
                                             l_row_ptrs.get_data()));
        const auto l_nnz =
            static_cast<size_type>(get_element(l_row_ptrs, size[0]));

        // create matrices
        auto l_mtx =
            Csr::create(exec, size, array<value_type>{exec, l_nnz},
                        array<index_type>{exec, l_nnz}, std::move(l_row_ptrs));

        // fill l_mtx with 1/w * (D + wL)
        exec->run(make_initialize_weighted_l(
            csr_matrix.get(), parameters_.relaxation_factor, l_mtx.get()));

        // invert the triangular matrices with triangular solvers
        auto l_trs = l_trs_factory->generate(std::move(l_mtx));

        return composition_type::create(std::move(l_trs));
    }
}


#define GKO_DECLARE_SOR(ValueType, IndexType) class Sor<ValueType, IndexType>

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SOR);


}  // namespace preconditioner
}  // namespace gko
