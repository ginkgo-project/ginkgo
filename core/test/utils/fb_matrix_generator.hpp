// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_FB_MATRIX_GENERATOR_HPP_
#define GKO_CORE_TEST_UTILS_FB_MATRIX_GENERATOR_HPP_


#include <numeric>
#include <random>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/unsort_matrix.hpp"


namespace gko {
namespace test {


/**
 * Generates a random matrix, ensuring the existence of diagonal entries.
 *
 * @tparam MatrixType  type of matrix to generate (matrix::Dense must implement
 *                     the interface `ConvertibleTo<MatrixType>`)
 * @tparam NonzeroDistribution  type of nonzero distribution
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 * @tparam MatrixArgs  the arguments from the matrix to be forwarded.
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param nonzero_dist  distribution of nonzeros per row
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 *
 * @return the unique pointer to generated matrix of type MatrixType
 */
template <typename MatrixType = matrix::Dense<>, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_random_matrix_with_diag(
    typename MatrixType::index_type num_rows,
    typename MatrixType::index_type num_cols,
    NonzeroDistribution&& nonzero_dist, ValueDistribution&& value_dist,
    Engine&& engine, std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;

    matrix_data<value_type, index_type> data{gko::dim<2>(num_rows, num_cols),
                                             {}};

    for (index_type row = 0; row < num_rows; ++row) {
        std::vector<index_type> col_idx(num_cols);
        std::iota(col_idx.begin(), col_idx.end(), size_type(0));
        // randomly generate number of nonzeros in this row
        auto nnz_in_row = static_cast<index_type>(nonzero_dist(engine));
        nnz_in_row = std::max(1, std::min(nnz_in_row, num_cols));
        // select a subset of `nnz_in_row` column indexes, and fill these
        // locations with random values
        std::shuffle(col_idx.begin(), col_idx.end(), engine);
        // add diagonal if it does not exist
        auto it = std::find(col_idx.begin(), col_idx.begin() + nnz_in_row, row);
        if (it == col_idx.begin() + nnz_in_row) {
            col_idx[nnz_in_row - 1] = row;
        }
        std::for_each(
            begin(col_idx), begin(col_idx) + nnz_in_row, [&](index_type col) {
                data.nonzeros.emplace_back(
                    row, col,
                    detail::get_rand_value<value_type>(value_dist, engine));
            });
    }

    data.sort_row_major();

    // convert to the correct matrix type
    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(data);
    return result;
}


/**
 * Generates a block CSR matrix having the same sparsity pattern as
 * a given CSR matrix.
 *
 * @param exec  Reference executor.
 * @param csrmat  The CSR matrix to use for the block sparsity pattern of the
 *                generated FBCSR matrix.
 * @param block_size  Block size of output Fbcsr matrix.
 * @param row_diag_dominant  If true, a row-diagonal-dominant Fbcsr matrix is
 *                           generated. Note that in this case, the input Csr
 *                           matrix must have diagonal entries in all rows.
 * @param rand_engine  Random number engine to use, such as
 * std::default_random_engine.
 */
template <typename ValueType, typename IndexType, typename RandEngine>
std::unique_ptr<matrix::Fbcsr<ValueType, IndexType>> generate_fbcsr_from_csr(
    const std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const csrmat, const int block_size,
    const bool row_diag_dominant, RandEngine&& rand_engine)
{
    const auto nbrows = static_cast<IndexType>(csrmat->get_size()[0]);
    const auto nbcols = static_cast<IndexType>(csrmat->get_size()[1]);
    const auto nbnz_temp =
        static_cast<IndexType>(csrmat->get_num_stored_elements());
    const int bs2 = block_size * block_size;

    auto fmtx = matrix::Fbcsr<ValueType, IndexType>::create(
        exec,
        dim<2>{static_cast<size_type>(nbrows * block_size),
               static_cast<size_type>(nbcols * block_size)},
        nbnz_temp * bs2, block_size);
    exec->copy(nbrows + 1, csrmat->get_const_row_ptrs(), fmtx->get_row_ptrs());
    exec->copy(nbnz_temp, csrmat->get_const_col_idxs(), fmtx->get_col_idxs());

    // We assume diagonal blocks are present for the diagonally-dominant case

    const IndexType nbnz = fmtx->get_num_stored_elements() / bs2;

    const IndexType* const row_ptrs = fmtx->get_const_row_ptrs();
    const IndexType* const col_idxs = fmtx->get_const_col_idxs();
    ValueType* const vals = fmtx->get_values();
    std::uniform_real_distribution<gko::remove_complex<ValueType>>
        off_diag_dist(-1.0, 1.0);

    for (IndexType ibrow = 0; ibrow < nbrows; ibrow++) {
        if (row_diag_dominant) {
            const IndexType nrownz =
                (row_ptrs[ibrow + 1] - row_ptrs[ibrow]) * block_size;

            std::uniform_real_distribution<gko::remove_complex<ValueType>>
                diag_dist(1.01 * nrownz, 2 * nrownz);

            for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
                 ibz++) {
                for (int i = 0; i < block_size * block_size; i++) {
                    vals[ibz * bs2 + i] =
                        gko::test::detail::get_rand_value<ValueType>(
                            off_diag_dist, rand_engine);
                }
                if (col_idxs[ibz] == ibrow) {
                    for (int i = 0; i < block_size; i++) {
                        vals[ibz * bs2 + i * block_size + i] =
                            static_cast<ValueType>(pow(-1, i)) *
                            gko::test::detail::get_rand_value<ValueType>(
                                diag_dist, rand_engine);
                    }
                }
            }
        } else {
            for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
                 ibz++) {
                for (int i = 0; i < bs2; i++) {
                    vals[ibz * bs2 + i] =
                        gko::test::detail::get_rand_value<ValueType>(
                            off_diag_dist, rand_engine);
                }
            }
        }
    }

    return fmtx;
}


/**
 * Generates a random block CSR matrix.
 *
 * The block sparsity pattern of the generated matrix always has the diagonal
 * entry in each block-row.
 *
 * @param exec  Reference executor.
 * @param nbrows  The number of block-rows in the generated matrix.
 * @param nbcols  The number of block-columns in the generated matrix.
 * @param mat_blk_sz  Block size of the generated matrix.
 * @param diag_dominant  If true, a row-diagonal-dominant Fbcsr matrix is
 *                       generated.
 * @param unsort  If true, the blocks of the generated matrix within each
 *                block-row are ordered randomly. Otherwise, blocks in each row
 *                are ordered by block-column index.
 * @param engine  Random number engine to use, such as
 * std::default_random_engine.
 */
template <typename ValueType, typename IndexType, typename RandEngine>
std::unique_ptr<matrix::Fbcsr<ValueType, IndexType>> generate_random_fbcsr(
    std::shared_ptr<const ReferenceExecutor> ref, const IndexType nbrows,
    const IndexType nbcols, const int mat_blk_sz, const bool diag_dominant,
    const bool unsort, RandEngine&& engine)
{
    using real_type = gko::remove_complex<ValueType>;
    std::unique_ptr<matrix::Csr<ValueType, IndexType>> rand_csr_ref =
        diag_dominant
            ? generate_random_matrix_with_diag<
                  matrix::Csr<ValueType, IndexType>>(
                  nbrows, nbcols,
                  std::uniform_int_distribution<IndexType>(0, nbcols - 1),
                  std::normal_distribution<real_type>(0.0, 1.0),
                  std::move(engine), ref)
            : generate_random_matrix<matrix::Csr<ValueType, IndexType>>(
                  nbrows, nbcols,
                  std::uniform_int_distribution<IndexType>(0, nbcols - 1),
                  std::normal_distribution<real_type>(0.0, 1.0),
                  std::move(engine), ref);
    if (unsort && rand_csr_ref->is_sorted_by_column_index()) {
        unsort_matrix(rand_csr_ref, engine);
    }
    return generate_fbcsr_from_csr(ref, rand_csr_ref.get(), mat_blk_sz,
                                   diag_dominant, std::move(engine));
}


}  // namespace test
}  // namespace gko

#endif  // GKO_CORE_TEST_UTILS_FB_MATRIX_GENERATOR_HPP_
