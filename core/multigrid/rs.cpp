// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/rs.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/multigrid/rs_kernels.hpp"


namespace gko {
namespace multigrid {
namespace rs {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);

GKO_REGISTER_OPERATION(check_m_matrix, rs::check_m_matrix);
GKO_REGISTER_OPERATION(compute_soc_and_run_rs, rs::compute_soc_and_run_rs);
GKO_REGISTER_OPERATION(fill_coarse_and_compute_prolong_row_ptrs,
                       rs::fill_coarse_and_compute_prolong_row_ptrs);
GKO_REGISTER_OPERATION(compute_interpolation, rs::compute_interpolation);

}  // anonymous namespace
}  // namespace rs


template <typename ValueType, typename IndexType>
void Rs<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;

    auto exec = this->get_executor();
    const auto fine_dim = this->system_matrix_->get_size()[0];

    const csr_type* rs_op = dynamic_cast<const csr_type*>(system_matrix_.get());
    std::shared_ptr<const csr_type> rs_op_shared_ptr{};

    if (!parameters_.skip_sorting || !rs_op) {
        rs_op_shared_ptr = convert_to_with_sorting<csr_type>(
            exec, system_matrix_, parameters_.skip_sorting);
        rs_op = rs_op_shared_ptr.get();
        this->set_fine_op(rs_op_shared_ptr);
    }
    array<bool> is_m_matrix_array(exec, 1);
    if (!parameters_.skip_m_matrix_check) {
        exec->run(rs::make_check_m_matrix(rs_op, is_m_matrix_array));
        if (!exec->copy_val_to_host(is_m_matrix_array.get_const_data())) {
            GKO_NOT_SUPPORTED(
                "RS coarsening requires an M-matrix (strictly positive "
                "diagonal, "
                "non-positive off-diagonals).");
        }
    }

    // define arrays
    array<bool> is_strong(exec, rs_op->get_num_stored_elements());
    array<IndexType> lambda(exec, fine_dim);
    array<IndexType> cf_marker(exec, fine_dim);
    IndexType coarse_dim{};
    // build Strength-of-Connection (SOC) mask, 1 byte per NNZ of the system
    // matrix, compute lambda, perform greedy RS C/F splitting:
    // 0 = undecided, 1 = C, -1 = F,
    // then extract coarse dims
    exec->run(rs::make_compute_soc_and_run_rs(
        rs_op, parameters_.strength_threshold, is_strong, lambda, cf_marker,
        coarse_dim));
    const size_type coarse_dim_size = static_cast<size_type>(coarse_dim);

    // fill in coarse_rows and fine_to_coarse, build prolongation using
    // interpolation
    array<IndexType> coarse_rows(exec, coarse_dim_size);
    array<IndexType> fine_to_coarse(exec, fine_dim);
    array<IndexType> prolong_row_ptrs(exec, fine_dim + 1);
    exec->run(rs::make_fill_coarse_and_compute_prolong_row_ptrs(
        cf_marker, coarse_rows, fine_to_coarse, rs_op, is_strong,
        prolong_row_ptrs));

    IndexType prolong_nnz =
        exec->copy_val_to_host(prolong_row_ptrs.get_const_data() + fine_dim);

    auto prolong_op = share(csr_type::create(
        exec, gko::dim<2>{fine_dim, coarse_dim_size},
        static_cast<size_type>(prolong_nnz), rs_op->get_strategy()));

    exec->copy_from(exec, fine_dim + 1, prolong_row_ptrs.get_const_data(),
                    prolong_op->get_row_ptrs());

    exec->run(rs::make_compute_interpolation(
        rs_op, is_strong.get_const_data(), cf_marker,
        fine_to_coarse.get_const_data(), prolong_op.get()));

    // build restriction as R = P^T
    auto restrict_op = share(as<csr_type>(prolong_op->transpose()));

    // coarse matrix (Ac = R  A  P)
    auto tmp = rs_op->multiply(prolong_op);
    tmp->set_strategy(rs_op->get_strategy());
    auto coarse_matrix = share(restrict_op->multiply(tmp));
    coarse_matrix->set_strategy(rs_op->get_strategy());

    this->set_multigrid_level(prolong_op, coarse_matrix, restrict_op);
}


template <typename ValueType, typename IndexType>
typename Rs<ValueType, IndexType>::parameters_type
Rs<ValueType, IndexType>::parse(const config::pnode& config,
                                const config::registry& context,
                                const config::type_descriptor& td_for_child)
{
    auto params = Rs<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("strength_threshold")) {
        params.with_strength_threshold(config::get_value<double>(obj));
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("skip_m_matrix_check")) {
        params.with_skip_m_matrix_check(config::get_value<bool>(obj));
    }

    return params;
}


#define GKO_DECLARE_RS(_vtype, _itype) class Rs<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RS);

}  // namespace multigrid
}  // namespace gko
