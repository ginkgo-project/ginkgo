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
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/rs_kernels.hpp"


namespace gko {
namespace multigrid {
namespace rs {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);

GKO_REGISTER_OPERATION(compute_soc_row_ptrs, rs::compute_soc_row_ptrs);
GKO_REGISTER_OPERATION(fill_soc, rs::fill_soc);
GKO_REGISTER_OPERATION(compute_lambda, rs::compute_lambda);
GKO_REGISTER_OPERATION(init_cf, rs::init_cf);
GKO_REGISTER_OPERATION(rs_coarsening, rs::rs_coarsening);
GKO_REGISTER_OPERATION(rs_cleanup, rs::rs_cleanup);
GKO_REGISTER_OPERATION(fill_coarse_rows, rs::fill_coarse_rows);
GKO_REGISTER_OPERATION(count_coarse, rs::count_coarse);


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

    // build Strength-of-Connection (SOC) – phase 1: row_ptrs + nnz
    array<IndexType> soc_row_ptrs(exec, fine_dim + 1);

    exec->run(rs::make_compute_soc_row_ptrs(
        rs_op, parameters_.strength_threshold, soc_row_ptrs.get_data()));

    IndexType soc_nnz =
        exec->copy_val_to_host(soc_row_ptrs.get_const_data() + fine_dim);
    const size_type soc_nnz_size = static_cast<size_type>(soc_nnz);

    auto soc = csr_type::create(exec, rs_op->get_size(), soc_nnz_size);
    soc->set_strategy(rs_op->get_strategy());

    exec->copy_from(exec, fine_dim + 1, soc_row_ptrs.get_const_data(),
                    soc->get_row_ptrs());

    // phase 2: fill col_idxs and values
    exec->run(
        rs::make_fill_soc(rs_op, parameters_.strength_threshold, soc.get()));

    // compute lambda
    array<IndexType> lambda(exec, fine_dim);
    exec->run(rs::make_compute_lambda(soc.get(), lambda.get_data()));

    // greedy RS C/F splitting: 0 = undecided, 1 = C, -1 = F
    array<IndexType> cf_marker(exec, fine_dim);
    exec->run(rs::make_init_cf(cf_marker));
    exec->run(rs::make_rs_coarsening(soc.get(), lambda.get_data(), cf_marker));
    exec->run(rs::make_rs_cleanup(cf_marker));

    // extract coarse rows
    IndexType coarse_dim{};
    exec->run(rs::make_count_coarse(cf_marker, &coarse_dim));
    const size_type coarse_dim_size = static_cast<size_type>(coarse_dim);

    array<IndexType> coarse_rows(exec, coarse_dim_size);
    exec->run(rs::make_fill_coarse_rows(cf_marker, coarse_rows.get_data()));

    // build restriction
    auto restrict_op =
        share(csr_type::create(exec, gko::dim<2>{coarse_dim_size, fine_dim},
                               coarse_dim_size, rs_op->get_strategy()));

    exec->copy_from(coarse_rows.get_executor(), coarse_dim_size,
                    coarse_rows.get_const_data(), restrict_op->get_col_idxs());

    exec->run(rs::make_fill_array(restrict_op->get_values(), coarse_dim_size,
                                  one<ValueType>()));
    exec->run(rs::make_fill_seq_array(restrict_op->get_row_ptrs(),
                                      coarse_dim_size + 1));

    auto prolong_op = gko::as<csr_type>(share(restrict_op->transpose()));

    //
    auto coarse_matrix = share(
        csr_type::create(exec, gko::dim<2>{coarse_dim_size, coarse_dim_size}));
    coarse_matrix->set_strategy(rs_op->get_strategy());

    auto tmp = csr_type::create(exec, gko::dim<2>{fine_dim, coarse_dim_size});
    tmp->set_strategy(rs_op->get_strategy());

    rs_op->apply(prolong_op, tmp);
    restrict_op->apply(tmp, coarse_matrix);

    this->set_multigrid_level(prolong_op, coarse_matrix, restrict_op);
}


#define GKO_DECLARE_RS(_vtype, _itype) class Rs<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RS);


}  // namespace multigrid
}  // namespace gko
