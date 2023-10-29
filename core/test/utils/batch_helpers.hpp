/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_CORE_TEST_UTILS_BATCH_HELPERS_HPP_
#define GKO_CORE_TEST_UTILS_BATCH_HELPERS_HPP_


#include <random>
#include <vector>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/log/batch_logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/assertions.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/utils/matrix_utils.hpp"


namespace gko {
namespace test {


/**
 * Converts a vector of unique pointers to a vector of shared pointers.
 */
template <typename T>
std::vector<std::shared_ptr<T>> share(std::vector<std::unique_ptr<T>>&& objs)
{
    std::vector<std::shared_ptr<T>> out;
    out.reserve(objs.size());
    for (auto& obj : objs) {
        out.push_back(std::move(obj));
    }
    return out;
}


/**
 * Generates a batch of random matrices of the specified type.
 */
template <typename MatrixType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_random_batch_matrix(
    const size_type num_batch_items, const size_type num_rows,
    const size_type num_cols, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine,
    std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    auto result = MatrixType::create(
        exec, batch_dim<2>(num_batch_items, dim<2>(num_rows, num_cols)),
        std::forward<MatrixArgs>(args)...);
    auto sp_mat = generate_random_device_matrix_data<value_type, index_type>(
        num_rows, num_cols, nonzero_dist, value_dist, engine,
        exec->get_master());
    auto row_idxs = gko::array<index_type>::const_view(
                        exec->get_master(), sp_mat.get_num_elems(),
                        sp_mat.get_const_row_idxs())
                        .copy_to_array();
    auto col_idxs = gko::array<index_type>::const_view(
                        exec->get_master(), sp_mat.get_num_elems(),
                        sp_mat.get_const_col_idxs())
                        .copy_to_array();

    for (size_type b = 0; b < num_batch_items; b++) {
        auto rand_mat =
            fill_random_matrix<typename MatrixType::unbatch_type, index_type>(
                num_rows, num_cols, row_idxs, col_idxs, value_dist, engine,
                exec);
        result->create_view_for_item(b)->copy_from(rand_mat.get());
    }

    return result;
}


/**
 * Generate a batch of 1D Poisson (3pt stencil, {-1, 5, -1}) matrices in the
 * given input matrix format.
 *
 * @tparam MatrixType  The concrete type of the output matrix.
 *
 * @param exec  The executor.
 * @param num_rows  The size (number of rows) of the generated matrix
 * @param num_batch_items  The number of Poisson matrices in the batch
 * @param args The create args to be forwarded to the matrix
 */
template <typename MatrixType, typename... MatrixArgs>
std::unique_ptr<const MatrixType> generate_3pt_stencil_batch_matrix(
    std::shared_ptr<const Executor> exec, const size_type num_batch_items,
    const int num_rows, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    const int num_cols = num_rows;
    gko::matrix_data<value_type, index_type> data{
        gko::dim<2>{static_cast<size_type>(num_rows),
                    static_cast<size_type>(num_cols)},
        {}};
    for (int row = 0; row < num_rows; ++row) {
        if (row > 0) {
            data.nonzeros.emplace_back(row - 1, row, value_type{-1.0});
        }
        data.nonzeros.emplace_back(row, row, value_type{5.0});
        if (row < num_rows - 1) {
            data.nonzeros.emplace_back(row, row + 1, value_type{-1.0});
        }
    }

    std::vector<gko::matrix_data<value_type, index_type>> batch_data(
        num_batch_items, data);
    return gko::batch::read<value_type, index_type, MatrixType>(
        exec, batch_data, std::forward<MatrixArgs>(args)...);
}


template <typename MatrixType, typename... MatrixArgs>
std::unique_ptr<const MatrixType> generate_diag_dominant_batch_matrix(
    std::shared_ptr<const gko::Executor> exec, const size_type num_batch_items,
    const int num_rows, const bool is_hermitian, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    using real_type = remove_complex<value_type>;
    using unbatch_type = typename MatrixType::unbatch_type;
    using multi_vec = batch::MultiVector<value_type>;
    using real_vec = batch::MultiVector<real_type>;
    const int num_cols = num_rows;
    gko::matrix_data<value_type, index_type> data{
        gko::dim<2>{static_cast<size_type>(num_rows),
                    static_cast<size_type>(num_cols)},
        {}};
    auto engine = std::default_random_engine(42);
    auto rand_diag_dist = std::normal_distribution<real_type>(4.0, 12.0);
    for (int row = 0; row < num_rows; ++row) {
        std::uniform_int_distribution<index_type> rand_nnz_dist{1, row + 1};
        const auto k = rand_nnz_dist(engine);
        if (row > 0) {
            data.nonzeros.emplace_back(row - 1, row, value_type{-1.0});
        }
        data.nonzeros.emplace_back(
            row, row,
            static_cast<value_type>(
                detail::get_rand_value<real_type>(rand_diag_dist, engine)));
        if (row < num_rows - 1) {
            data.nonzeros.emplace_back(row, k, value_type{-1.0});
            data.nonzeros.emplace_back(row, row + 1, value_type{-1.0});
        }
    }

    if (is_hermitian) {
        gko::utils::make_hpd(data);
    }
    data.ensure_row_major_order();

    auto soa_data =
        gko::device_matrix_data<value_type, index_type>::create_from_host(
            exec->get_master(), data);
    auto row_idxs = gko::array<index_type>::const_view(
                        exec->get_master(), soa_data.get_num_elems(),
                        soa_data.get_const_row_idxs())
                        .copy_to_array();
    auto col_idxs = gko::array<index_type>::const_view(
                        exec->get_master(), soa_data.get_num_elems(),
                        soa_data.get_const_col_idxs())
                        .copy_to_array();

    std::vector<gko::matrix_data<value_type, index_type>> batch_data;
    batch_data.reserve(num_batch_items);
    batch_data.emplace_back(data);
    auto rand_val_dist = std::normal_distribution<>(-0.5, 0.5);
    for (size_type b = 1; b < num_batch_items; b++) {
        auto rand_data = fill_random_matrix_data<value_type, index_type>(
            num_rows, num_cols, row_idxs, col_idxs, rand_val_dist, engine);
        gko::utils::make_diag_dominant(rand_data);
        batch_data.emplace_back(rand_data);
        GKO_ASSERT(rand_data.size == batch_data.at(0).size);
    }
    return gko::batch::read<value_type, index_type, MatrixType>(
        exec, batch_data, std::forward<MatrixArgs>(args)...);
}


template <typename MatrixType>
struct LinearSystem {
    using value_type = typename MatrixType::value_type;
    using multi_vec = batch::MultiVector<value_type>;
    using real_vec = batch::MultiVector<remove_complex<value_type>>;

    std::shared_ptr<const MatrixType> matrix;
    std::shared_ptr<multi_vec> rhs;
    std::shared_ptr<real_vec> rhs_norm;
    std::shared_ptr<multi_vec> exact_sol;
};


template <typename MatrixType>
LinearSystem<MatrixType> generate_batch_linear_system(
    std::shared_ptr<const MatrixType> input_batch_matrix, const int num_rhs)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    using multi_vec = batch::MultiVector<value_type>;
    using real_vec = batch::MultiVector<remove_complex<value_type>>;
    LinearSystem<MatrixType> sys;
    sys.matrix = input_batch_matrix;
    const auto num_batch_items = sys.matrix->get_num_batch_items();
    const auto num_rows = sys.matrix->get_common_size()[0];
    auto exec = sys.matrix->get_executor();
    sys.exact_sol = multi_vec::create(
        exec, batch_dim<2>(num_batch_items, gko::dim<2>(num_rows, num_rhs)));
    sys.exact_sol->fill(value_type{2.0});

    sys.rhs = multi_vec::create_with_config_of(sys.exact_sol);
    // A * x_{exact} = b
    sys.matrix->apply(sys.exact_sol, sys.rhs);
    const gko::batch_dim<2> norm_dim(num_batch_items, gko::dim<2>(1, num_rhs));
    sys.rhs_norm = real_vec::create(exec, norm_dim);
    sys.rhs->compute_norm2(sys.rhs_norm.get());
    return sys;
}


template <typename MatrixType>
std::unique_ptr<
    batch::MultiVector<remove_complex<typename MatrixType::value_type>>>
compute_residual_norms(
    const MatrixType* mtx,
    const batch::MultiVector<typename MatrixType::value_type>* b,
    const batch::MultiVector<typename MatrixType::value_type>* x)
{
    using value_type = typename MatrixType::value_type;
    using multi_vec = batch::MultiVector<value_type>;
    using real_vec = batch::MultiVector<remove_complex<value_type>>;
    auto exec = mtx->get_executor();
    auto num_batch_items = x->get_num_batch_items();
    auto num_rhs = x->get_common_size()[1];
    const gko::batch_dim<2> norm_dim(num_batch_items, gko::dim<2>(1, num_rhs));

    auto residual_vec = b->clone();
    auto res_norms = real_vec::create(exec, norm_dim);
    auto alpha =
        gko::batch::initialize<multi_vec>(num_batch_items, {-1.0}, exec);
    auto beta = gko::batch::initialize<multi_vec>(num_batch_items, {1.0}, exec);
    mtx->apply(alpha, x, beta, residual_vec);
    residual_vec->compute_norm2(res_norms);
    return res_norms;
}


template <typename ValueType>
struct Result {
    using multi_vec = batch::MultiVector<ValueType>;
    using real_vec = batch::MultiVector<remove_complex<ValueType>>;

    std::shared_ptr<multi_vec> x;
    std::shared_ptr<real_vec> res_norm;
};


template <typename ValueType>
struct ResultWithLogData : public Result<ValueType> {
    std::unique_ptr<
        gko::batch::log::detail::log_data<remove_complex<ValueType>>>
        log_data;
};


template <typename MatrixType, typename SolverType>
Result<typename MatrixType::value_type> solve_linear_system(
    std::shared_ptr<const Executor> exec, const LinearSystem<MatrixType>& sys,
    std::shared_ptr<SolverType> solver)
{
    using value_type = typename MatrixType::value_type;
    using real_type = remove_complex<value_type>;
    using multi_vec = typename Result<value_type>::multi_vec;
    using real_vec = typename Result<value_type>::real_vec;

    const size_type num_batch_items = sys.matrix->get_num_batch_items();
    const int num_rows = sys.matrix->get_common_size()[0];
    const int num_rhs = sys.rhs->get_common_size()[1];
    const gko::batch_dim<2> vec_size(num_batch_items,
                                     gko::dim<2>(num_rows, num_rhs));
    const gko::batch_dim<2> norm_size(num_batch_items, gko::dim<2>(1, num_rhs));

    Result<value_type> result;
    result.x = multi_vec::create_with_config_of(sys.rhs);
    result.x->fill(zero<value_type>());

    solver->apply(sys.rhs, result.x);
    result.res_norm =
        compute_residual_norms(sys.matrix.get(), sys.rhs.get(), result.x.get());

    return std::move(result);
}


template <typename MatrixType, typename SolveLambda, typename Settings>
ResultWithLogData<typename MatrixType::value_type> solve_linear_system(
    std::shared_ptr<const Executor> exec, SolveLambda solve_lambda,
    const Settings settings, const LinearSystem<MatrixType>& sys,
    std::shared_ptr<batch::BatchLinOpFactory> precond_factory = nullptr)
{
    using value_type = typename MatrixType::value_type;
    using real_type = remove_complex<value_type>;
    using multi_vec = typename Result<value_type>::multi_vec;
    using real_vec = typename Result<value_type>::real_vec;

    const size_type num_batch_items = sys.matrix->get_num_batch_items();
    const int num_rows = sys.matrix->get_common_size()[0];
    const int num_rhs = sys.rhs->get_common_size()[1];
    const gko::batch_dim<2> norm_size(num_batch_items, gko::dim<2>(1, num_rhs));

    ResultWithLogData<value_type> result;
    result.x = multi_vec::create_with_config_of(sys.rhs);
    result.x->fill(zero<value_type>());

    auto log_data = std::make_unique<batch::log::detail::log_data<real_type>>(
        exec, num_batch_items);

    std::unique_ptr<gko::batch::BatchLinOp> precond;
    if (precond_factory) {
        precond = precond_factory->generate(sys.matrix);
    } else {
        precond = nullptr;
    }

    solve_lambda(settings, precond.get(), sys.matrix.get(), sys.rhs.get(),
                 result.x.get(), *log_data.get());


    result.log_data = std::make_unique<batch::log::detail::log_data<real_type>>(
        exec->get_master(), num_batch_items);
    result.log_data->iter_counts = log_data->iter_counts;
    result.log_data->res_norms = log_data->res_norms;

    result.res_norm =
        compute_residual_norms(sys.matrix.get(), sys.rhs.get(), result.x.get());

    return std::move(result);
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_BATCH_HELPERS_HPP_
