// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iomanip>
#include <iostream>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/iterator_factory.hpp"

namespace gko {
namespace experimental {
namespace eigensolver {

template <typename ValueType>
void pretty_print(std::shared_ptr<gko::matrix::Dense<ValueType>> matrix,
                  std::string name = "")
{
    std::cout << "========" << name << "========" << std::endl;
    for (int row = 0; row < matrix->get_size()[0]; row++) {
        for (int col = 0; col < matrix->get_size()[1]; col++) {
            std::cout << std::setw(5) << std::setprecision(3)
                      << matrix->at(row, col) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "================" << std::endl;
}

template <typename ValueType>
void eigen(std::shared_ptr<gko::matrix::Dense<ValueType>> matrix,
           std::shared_ptr<gko::matrix::Dense<ValueType>> eigenvector)
{
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "solving small" << std::endl;
    pretty_print(matrix, "original_small_matrix");
    std::cout << "--------------------------------------" << std::endl;
    if (matrix->get_size()[0] == 1) {
        eigenvector->at(0, 0) = one<ValueType>();
    } else if (matrix->get_size()[0] == 2) {
        auto eps = 1e-14;
        // when the matrix is diagonal (up to machine precision)
        if (abs(matrix->at(0, 1)) <=
            eps * (abs(matrix->at(0, 0)) + abs(matrix->at(1, 1)))) {
            // set the off diagonal to zero
            matrix->at(0, 1) = zero<ValueType>();
            matrix->at(1, 0) = zero<ValueType>();
            // eigenvector is identity
            eigenvector->at(0, 0) = one<ValueType>();
            eigenvector->at(0, 1) = zero<ValueType>();
            eigenvector->at(1, 0) = zero<ValueType>();
            eigenvector->at(1, 1) = one<ValueType>();
            return;
        } else if (abs(matrix->at(0, 0) - matrix->at(1, 1)) <=
                   eps * (abs(matrix->at(0, 0)) + abs(matrix->at(1, 1)))) {
            // when the diagonal values are identical
            // we swap the value to make it ascending
            matrix->at(0, 0) -= matrix->at(0, 1);
            matrix->at(1, 1) += matrix->at(1, 0);
            matrix->at(0, 1) = zero<ValueType>();
            matrix->at(1, 0) = zero<ValueType>();
            ValueType temp = one<ValueType>() / sqrt(2.0);
            eigenvector->at(0, 0) = -temp;
            eigenvector->at(0, 1) = temp;
            eigenvector->at(1, 0) = temp;
            eigenvector->at(1, 1) = temp;
            pretty_print(matrix, "eigenvalue by second");
            pretty_print(eigenvector, "eigenvector by second");
            return;
        }
        // general case, diagonalized by [c s; -s c] -> tau(2\theta) = 2 * A_01
        // / (A_00-A_11) todo: check
        ValueType cot2 =
            (matrix->at(1, 1) - matrix->at(0, 0)) / (2 * matrix->at(0, 1));
        ValueType tan =
            one<ValueType>() /
            (cot2 + (cot2 >= zero<ValueType>() ? one<ValueType>()
                                               : -one<ValueType>()) *
                        sqrt(one<ValueType>() + cot2 * cot2));
        ValueType cos = one<ValueType>() / sqrt(one<ValueType>() + tan * tan);
        ValueType sin = tan * cos;
        eigenvector->at(0, 0) = cos;
        eigenvector->at(0, 1) = sin;
        eigenvector->at(1, 0) = -sin;
        eigenvector->at(1, 1) = cos;
        ValueType lambda_1 = cos * cos * matrix->at(0, 0) -
                             2.0 * cos * sin * matrix->at(0, 1) +
                             sin * sin * matrix->at(1, 1);
        ValueType lambda_2 = sin * sin * matrix->at(0, 0) +
                             2.0 * cos * sin * matrix->at(0, 1) +
                             cos * cos * matrix->at(1, 1);
        matrix->at(0, 0) = lambda_1;
        matrix->at(1, 1) = lambda_2;
        matrix->at(0, 1) = zero<ValueType>();
        matrix->at(1, 0) = zero<ValueType>();
        if (lambda_1 > lambda_2) {
            matrix->at(0, 0) = lambda_2;
            matrix->at(1, 1) = lambda_1;
            eigenvector->at(0, 0) = sin;
            eigenvector->at(0, 1) = cos;
            eigenvector->at(1, 0) = cos;
            eigenvector->at(1, 1) = -sin;
        }
        pretty_print(matrix, "eigenvalue by general");
        pretty_print(eigenvector, "eigenvector by general");
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void dense_permute(std::shared_ptr<gko::matrix::Dense<ValueType>> input,
                   const gko::array<int>& permuted,
                   std::shared_ptr<gko::matrix::Dense<ValueType>> output)
{
    for (int row = 0; row < input->get_size()[0]; row++) {
        for (int col = 0; col < input->get_size()[1]; col++) {
            output->at(row, col) =
                input->at(row, permuted.get_const_data()[col]);
        }
    }
}

template <typename ValueType>
void diag_permute(std::shared_ptr<gko::matrix::Dense<ValueType>> input,
                  const gko::array<int>& permuted,
                  std::shared_ptr<gko::matrix::Dense<ValueType>> output)
{
    for (int row = 0; row < input->get_size()[0]; row++) {
        output->at(row, row) = input->at(permuted.get_const_data()[row],
                                         permuted.get_const_data()[row]);
    }
}


template <typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>> jacobi(
    std::shared_ptr<gko::matrix::Dense<ValueType>> matrix, int block_size,
    double tol, int max_iter)
{
    auto exec = matrix->get_executor();
    auto small_eigen_size = 2 * block_size;
    GKO_THROW_IF_INVALID(
        matrix->get_size()[0] % small_eigen_size == 0,
        "2 * blocksize must be devided by the matrix size now");

    // number of parallel problems to be solved simultaneously
    auto num_prob_per_stage = matrix->get_size()[0] / small_eigen_size;
    // // initial partitioning (0,1), (2,3), (4,5), ...
    std::vector<std::pair<int, int>> coord(num_prob_per_stage);
    for (int i = 0; i < num_prob_per_stage; i++) {
        coord.at(i) = std::make_pair(2 * i, 2 * i + 1);
    }

    // initial eigenvector matrix = I
    auto eigenvector = gko::share(
        gko::matrix::Dense<ValueType>::create(exec, matrix->get_size()));
    eigenvector->fill(0.0);
    for (size_t i = 0; i < eigenvector->get_size()[0]; i++) {
        eigenvector->at(i, i) = one<ValueType>();
    }

    double norm_diagonal = 0.0;
    double norm_off_diagonal = 0.0;
    for (size_t row = 0; row < matrix->get_size()[0]; row++) {
        for (size_t col = 0; col < matrix->get_size()[1]; col++) {
            if (row == col) {
                norm_diagonal += matrix->at(row, col) * matrix->at(row, col);
            } else {
                norm_off_diagonal +=
                    matrix->at(row, col) * matrix->at(row, col);
            }
        }
    }

    int iter = 0;
    using dense = gko::matrix::Dense<ValueType>;
    std::vector<std::shared_ptr<dense>> local_problem_vector(
        num_prob_per_stage);
    std::vector<std::shared_ptr<dense>> local_eigen_vector(num_prob_per_stage);
    std::vector<std::shared_ptr<dense>> temp_vector(num_prob_per_stage);
    std::vector<array<ValueType>> space_vector;
    std::vector<array<ValueType>> large_space_vector;
    std::vector<array<ValueType>> original_diagonal_vector;
    std::vector<array<int>> sort_idx_vector;
    std::vector<array<int>> invert_idx_vector;
    space_vector.reserve(num_prob_per_stage);
    original_diagonal_vector.reserve(num_prob_per_stage);
    sort_idx_vector.reserve(num_prob_per_stage);
    invert_idx_vector.reserve(num_prob_per_stage);
    for (int i = 0; i < num_prob_per_stage; i++) {
        local_problem_vector.at(i) =
            dense::create(exec, dim<2>{small_eigen_size, small_eigen_size});
        local_eigen_vector.at(i) =
            dense::create(exec, dim<2>{small_eigen_size, small_eigen_size});
        temp_vector.at(i) =
            dense::create(exec, dim<2>{small_eigen_size, small_eigen_size});
        space_vector.emplace_back(exec, block_size * matrix->get_size()[1]);
        large_space_vector.emplace_back(
            exec, small_eigen_size * matrix->get_size()[1]);
        original_diagonal_vector.emplace_back(exec, small_eigen_size);
        sort_idx_vector.emplace_back(exec, small_eigen_size);
        invert_idx_vector.emplace_back(exec, small_eigen_size);
    }
    while (norm_off_diagonal > tol * norm_diagonal && iter < max_iter) {
        std::cout << "=======================" << std::endl;
        std::cout << "current history:" << norm_off_diagonal / norm_diagonal
                  << std::endl;
        std::cout << norm_off_diagonal << " " << norm_diagonal << std::endl;
        iter = iter + 1;
        // compute a sweep
        for (size_t i = 0; i < 2 * num_prob_per_stage - 1; i++) {
            std::cout << "+++++++++++++ stage: " << i << "+++++++++++++"
                      << std::endl;
            // this loop could in principle be performed in parallel
            // but multiplication of D must be synchronized, at first
            // multiplication from the right with VV is done in common
            // after that, when all multiplications D(:,[I J])*VV are
            // finished, then all multiplications VV'*D([I J],:) can
            // be computed
            // first stage: solve small eigenvalue problem and update norms
            double mynrm_diagonal = 0.0;
            double mynrm_off_diagonal = 0.0;
            // first sub block  I=r*block_size,...,(r+1)*block_size-1
            // second sub block J=s*block_size,...,(s+1)*block_size-1
            // we only deal with block_size = 1 now
            std::cout << "first stage" << std::endl;
            for (size_t j = 0; j < num_prob_per_stage; j++) {
                auto local_problem = local_problem_vector.at(j);
                auto local_eigen = local_eigen_vector.at(j);
                auto temp = temp_vector.at(j);
                auto& original_diagonal = original_diagonal_vector.at(j);
                auto& sort_idx = sort_idx_vector.at(j);
                auto& invert_idx = invert_idx_vector.at(j);
                // solve eigenvalue problem of block (r, s) = coord(j)
                // first sub block  I=r*block_size,...,(r+1)*block_size-1
                // second sub block J=s*block_size,...,(s+1)*block_size-1
                // local symmetric matrix block DD=A([I J],[I J]);
                auto block_coord = coord.at(j);
                // 1. local(0:bs-1,0:bs-1)=A(I,I)
                // 2. local(0:bs-1,bs:end)=A(I,J)
                // 3. local(bs:end,0:bs-1)=A(J,I)
                // 4. local(bs:end,bs:end)=A(J,J)
                for (int row_block = 0; row_block < 2; row_block++) {
                    auto row_block_idx =
                        row_block == 0 ? block_coord.first : block_coord.second;
                    for (int col_block = 0; col_block < 2; col_block++) {
                        auto col_block_idx = col_block == 0
                                                 ? block_coord.first
                                                 : block_coord.second;
                        // TODO: replaced by submatrix copy_from or maybe we can
                        // operate on the original matrix directly?
                        for (size_t row = 0; row < block_size; row++) {
                            for (size_t col = 0; col < block_size; col++) {
                                local_eigen->at(row + row_block * block_size,
                                                col + col_block * block_size) =
                                    matrix->at(
                                        row_block_idx * block_size + row,
                                        col_block_idx * block_size + col);
                            }
                        }
                    }
                }
                // store the original diagoanl
                for (size_t row = 0; row < small_eigen_size; row++) {
                    original_diagonal.get_data()[row] =
                        local_eigen->at(row, row);
                }
                double norm_diagonal_local = 0.0;
                double norm_off_diagonal_local = 0.0;
                // local square sum of the diagonal entries
                // local square sum of the off-diagonal entries
                for (size_t row = 0; row < local_eigen->get_size()[0]; row++) {
                    for (size_t col = 0; col < local_eigen->get_size()[1];
                         col++) {
                        if (row == col) {
                            norm_diagonal_local += local_eigen->at(row, col) *
                                                   local_eigen->at(row, col);
                        } else {
                            norm_off_diagonal_local +=
                                local_eigen->at(row, col) *
                                local_eigen->at(row, col);
                        }
                    }
                }

                // locally downdate squared norm of the diagonal entries
                mynrm_diagonal -= norm_diagonal_local;
                // locally downdate squared norm of the off-diagonal entries
                mynrm_off_diagonal -= norm_off_diagonal_local;
                std::cout << "internal: " << mynrm_diagonal << " "
                          << mynrm_off_diagonal << std::endl;

                // TODO? any reason? copy diagonal entries of A([I J],[I J])
                // pA = AA[j];
                // pD = DD[j];
                // for (jj = 0; jj < k; jj++, pA++, pD += k + 1) *pA = *pD;

                // solve local eigenvalue problem
                // assume it contains eigenvalue in ascending order.
                eigen(local_eigen, temp);
                // find out which eigenvalues are closest to the diagonal
                // entries of AA. This is necessary to make sure that VV tends
                // to I when AA tends to be diagonal natural ordering
                std::iota(sort_idx.get_data(),
                          sort_idx.get_data() + small_eigen_size, int{});
                auto zip = detail::make_zip_iterator(
                    sort_idx.get_data(), original_diagonal.get_data());
                // sort diagonal entries in ascending order
                std::sort(zip, zip + small_eigen_size, [](auto lhs, auto rhs) {
                    return get<1>(lhs) < get<1>(rhs);
                });
                // invert the permutation
                for (size_t idx = 0; idx < small_eigen_size; idx++) {
                    invert_idx.get_data()[sort_idx.get_const_data()[idx]] = idx;
                }
                diag_permute(local_eigen, invert_idx, local_problem);
                dense_permute(temp, invert_idx, local_eigen);
                pretty_print(local_problem, "eigenvalue after sorted");
                pretty_print(local_eigen, "eigenvector after sorted");
                // write(std::cout, local_problem);

                // locally update squared norm of the diagonal entries
                norm_diagonal_local = 0.0;
                norm_off_diagonal_local = 0.0;
                for (size_t row = 0; row < small_eigen_size; row++) {
                    for (size_t col = 0; col < local_problem->get_size()[1];
                         col++) {
                        if (row == col) {
                            norm_diagonal_local += local_problem->at(row, col) *
                                                   local_problem->at(row, col);
                        } else {
                            norm_off_diagonal_local +=
                                local_problem->at(row, col) *
                                local_problem->at(row, col);
                        }
                    }
                }
                mynrm_diagonal += norm_diagonal_local;
                mynrm_off_diagonal += norm_off_diagonal_local;
                std::cout << "internal 2 -> " << norm_diagonal_local << " "
                          << norm_off_diagonal_local << std::endl;
                std::cout << "internal 3 -> " << mynrm_diagonal << " "
                          << mynrm_off_diagonal << std::endl;
            }
            // globally downdate squared norm of the diagonal entries
            norm_diagonal += mynrm_diagonal;
            // globally downdate squared norm of the off-diagonal entries
            norm_off_diagonal += mynrm_off_diagonal;
            std::cout << "second stage" << std::endl;
            // second stage: multiplication from the right
            for (size_t j = 0; j < num_prob_per_stage; j++) {
                auto block_coord = coord.at(j);
                // first sub block  I=r*block_size,...,(r+1)*block_size-1
                // second sub block J=s*block_size,...,(s+1)*block_size-1
                auto local_problem = local_problem_vector.at(j);
                auto local_eigen = local_eigen_vector.at(j);
                auto& space = space_vector.at(j);
                auto& large_space = large_space_vector.at(j);
                // eigenvector, matrix
                // update eigenvector matrix
                // V(:,[I J])=V(:,[I J])*VV{j};
                // V(:, I) = V(:, I) * VV{j}
                auto block = dense::create(
                    exec,
                    gko::dim<2>{eigenvector->get_size()[0], small_eigen_size},
                    large_space.as_view(), small_eigen_size);
                for (int col_block = 0; col_block < 2; col_block++) {
                    auto col_block_idx =
                        col_block == 0 ? block_coord.first : block_coord.second;
                    // TODO: replaced by submatrix copy_from or maybe we can
                    // operate on the original matrix directly?
                    for (size_t row = 0; row < eigenvector->get_size()[0];
                         row++) {
                        for (size_t col = 0; col < block_size; col++) {
                            block->at(row, col + col_block * block_size) =
                                eigenvector->at(
                                    row, col_block_idx * block_size + col);
                        }
                    }
                }

                auto update = local_eigen->create_submatrix(
                    {0, local_eigen->get_size()[0]}, {0, block_size});
                auto result = eigenvector->create_submatrix(
                    {0, eigenvector->get_size()[0]},
                    {block_coord.first * block_size,
                     (block_coord.first + 1) * block_size});
                block->apply(update, result);
                // V(:, J) = V(:, J) * VV{j}
                update = local_eigen->create_submatrix(
                    {0, local_eigen->get_size()[0]},
                    {block_size, 2 * block_size});
                result = eigenvector->create_submatrix(
                    {0, eigenvector->get_size()[0]},
                    {block_coord.second * block_size,
                     (block_coord.second + 1) * block_size});
                block->apply(update, result);
                // update eigenvalue matrix
                // A(:,[I J])=A(:,[I J])*VV{j};
                for (int col_block = 0; col_block < 2; col_block++) {
                    auto col_block_idx =
                        col_block == 0 ? block_coord.first : block_coord.second;
                    // TODO: replaced by submatrix copy_from or maybe we can
                    // operate on the original matrix directly?
                    for (size_t row = 0; row < matrix->get_size()[0]; row++) {
                        for (size_t col = 0; col < block_size; col++) {
                            block->at(row, col + col_block * block_size) =
                                matrix->at(row,
                                           col_block_idx * block_size + col);
                        }
                    }
                }
                update = local_eigen->create_submatrix(
                    {0, local_eigen->get_size()[0]}, {0, block_size});
                result = matrix->create_submatrix(
                    {0, matrix->get_size()[0]},
                    {block_coord.first * block_size,
                     (block_coord.first + 1) * block_size});
                block->apply(update, result);
                update = local_eigen->create_submatrix(
                    {0, local_eigen->get_size()[0]},
                    {block_size, 2 * block_size});
                result = matrix->create_submatrix(
                    {0, matrix->get_size()[0]},
                    {block_coord.second * block_size,
                     (block_coord.second + 1) * block_size});
                block->apply(update, result);
            }  // end for j
            // write(std::cout, matrix);
            // write(std::cout, eigenvector);
            std::cout << "third stage" << std::endl;
            // third stage: multiplication of A from the left
            for (size_t j = 0; j < num_prob_per_stage; j++) {
                auto block_coord = coord.at(j);
                // first sub block  I=r*block_size,...,(r+1)*block_size-1
                // second sub block J=s*block_size,...,(s+1)*block_size-1
                auto local_problem = local_problem_vector.at(j);
                auto local_eigen = local_eigen_vector.at(j);
                auto temp = temp_vector.at(j);
                auto& space = space_vector.at(j);
                auto& large_space = large_space_vector.at(j);
                // first sub block  I=r*block_size,...,(r+1)*block_size-1
                // second sub block J=s*block_size,...,(s+1)*block_size-1
                // update eigenvalue matrix
                // A([I J],:)=VV{j}'*A([I J],:);
                local_eigen->transpose(temp);
                // gko::write(std::cout, temp);
                auto block = dense::create(
                    exec, gko::dim<2>{small_eigen_size, matrix->get_size()[1]},
                    large_space.as_view(), matrix->get_size()[1]);
                for (int row_block = 0; row_block < 2; row_block++) {
                    auto row_block_idx =
                        row_block == 0 ? block_coord.first : block_coord.second;
                    // TODO: replaced by submatrix copy_from or maybe we can
                    // operate on the original matrix directly?
                    for (size_t row = 0; row < block_size; row++) {
                        for (size_t col = 0; col < matrix->get_size()[1];
                             col++) {
                            block->at(row + row_block * block_size, col) =
                                matrix->at(row_block_idx * block_size + row,
                                           col);
                        }
                    }
                }
                // write(std::cout, block);
                // TODO: I think column view should only available for one
                // vector?
                auto update = temp->create_submatrix({0, block_size},
                                                     {0, temp->get_size()[1]});
                auto result = matrix->create_submatrix(
                    {block_coord.first * block_size,
                     (block_coord.first + 1) * block_size},
                    {0, matrix->get_size()[1]});
                update->apply(block, result);
                update = temp->create_submatrix({block_size, 2 * block_size},
                                                {0, temp->get_size()[1]});
                result = matrix->create_submatrix(
                    {block_coord.second * block_size,
                     (block_coord.second + 1) * block_size},
                    {0, matrix->get_size()[1]});
                update->apply(block, result);
            }  // end for j

            // write(std::cout, matrix);
            std::cout << "forth stage" << std::endl;
            // fourth stage: diagonal blocks and new coordinates
            for (size_t j = 0; j < num_prob_per_stage; j++) {
                // update eigenvalue matrix from local to matrix (diagoanl =
                // eigenvalue, others are zero)
                auto block_coord = coord.at(j);
                auto local_problem = local_problem_vector.at(j);
                for (int row_block = 0; row_block < 2; row_block++) {
                    auto row_block_idx =
                        row_block == 0 ? block_coord.first : block_coord.second;
                    for (int col_block = 0; col_block < 2; col_block++) {
                        auto col_block_idx = col_block == 0
                                                 ? block_coord.first
                                                 : block_coord.second;
                        // TODO: replaced by submatrix copy_from or maybe we can
                        // operate on the original matrix directly?
                        for (size_t row = 0; row < block_size; row++) {
                            for (size_t col = 0; col < block_size; col++) {
                                if (row == col && row_block == col_block) {
                                    matrix->at(
                                        row_block_idx * block_size + row,
                                        col_block_idx * block_size + col) =
                                        local_problem->at(
                                            row + row_block * block_size,
                                            col + col_block * block_size);
                                } else {
                                    matrix->at(row_block_idx * block_size + row,
                                               col_block_idx * block_size +
                                                   col) = zero<ValueType>();
                                }
                            }
                        }
                    }
                }

                auto update_coord = [num_prob_per_stage](auto idx) {
                    // consider idx + 1
                    // 1 -> 1
                    // even -> try +2, if not, idx+1 = 2*num_prob_per_stage - 1
                    // odd -> try -2, if it is less than 3, idx + 1 = 2
                    if (idx + 1 != 1) {
                        // even case
                        if ((idx + 1) % 2 == 0) {
                            // increase by 2 if possible
                            if (idx + 1 < 2 * num_prob_per_stage) {
                                idx += 2;
                            } else {  // => idx+1=2*num_prob_per_stage
                                idx = 2 * num_prob_per_stage - 2;
                            }
                        } else {  // odd case
                            // decrease by 2 if possible
                            if (idx + 1 > 3) {
                                idx -= 2;
                            } else {  // => idx+1=3
                                idx = 1;
                            }
                        }
                    }
                    return idx;
                };
                // new coord for next sweep
                coord.at(j) = std::make_pair(update_coord(block_coord.first),
                                             update_coord(block_coord.second));
            }  // end for j
            for (size_t j = 0; j < num_prob_per_stage; j++) {
                std::cout << "(" << coord.at(j).first << ", "
                          << coord.at(j).second << ") ";
            }
            std::cout << std::endl;
            pretty_print(matrix, "after stage");
            pretty_print(eigenvector, "after stage");
        }  // end for i
    }      // end while
    // TODO: remove potential additional entries
    // TODO: reorder eigenvalue
    return eigenvector;
}

template std::shared_ptr<gko::matrix::Dense<float>> jacobi(
    std::shared_ptr<gko::matrix::Dense<float>> matrix, int block_size,
    double tol, int max_iter);
template std::shared_ptr<gko::matrix::Dense<double>> jacobi(
    std::shared_ptr<gko::matrix::Dense<double>> matrix, int block_size,
    double tol, int max_iter);
}  // namespace eigensolver
}  // namespace experimental
}  // namespace gko
