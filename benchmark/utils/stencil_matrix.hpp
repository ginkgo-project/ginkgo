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

#ifndef GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP
#define GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP


#include <ginkgo/ginkgo.hpp>


template <typename T>
double closest_nth_root(T v, int n)
{
    auto root = std::pow(v, 1. / static_cast<double>(n));
    auto root_floor = std::floor(root);
    auto root_ceil = std::ceil(root);
    if (root - root_floor > root_ceil - root) {
        return root_ceil;
    } else {
        return root_floor;
    }
}


/**
 * Generates matrix data for a 2D stencil matrix. If restricted is set to true,
 * creates a 5-pt stencil, if it is false creates a 9-pt stencil. If
 * strong_scaling is set to true, creates the same problem size independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_stencil(
    gko::mpi::communicator comm, const gko::size_type target_local_size,
    bool restricted)
{
    std::array<int, 2> dims{};
    std::array<int, 2> periods{};
    MPI_Dims_create(comm.size(), dims.size(), dims.data());

    MPI_Comm cart_comm;
    MPI_Cart_create(comm.get(), dims.size(), dims.data(), periods.data(), 0,
                    &cart_comm);

    std::array<int, 2> coords{};
    MPI_Cart_coords(cart_comm, comm.rank(), coords.size(), coords.data());

    const auto dp =
        static_cast<IndexType>(closest_nth_root(target_local_size, 2));
    const auto local_size = static_cast<gko::size_type>(dp * dp);
    const auto global_size = local_size * comm.size();
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{global_size, global_size});

    auto global_offset = [&](const int px, const int py) {
        return static_cast<int>(local_size) * px +
               static_cast<int>(local_size) * dims[0] * py;
    };

    auto target_coords = [&](const IndexType i, const int coord) {
        return 0 <= i && i <= dp - 1 ? coord : i < 0 ? coord - 1 : coord + 1;
    };

    auto target_local_idx = [&](const IndexType i) {
        return 0 <= i && i <= dp - 1 ? i : i < 0 ? dp + i : dp - i;
    };

    auto flat_idx = [&](const IndexType ix, IndexType iy) {
        return global_offset(target_coords(ix, coords[0]),
                             target_coords(iy, coords[1])) +
               target_local_idx(ix) + target_local_idx(iy) * dp;
    };

    auto is_valid_idx = [&](const IndexType i) {
        return i >= 0 && i < static_cast<IndexType>(global_size);
    };

    auto nnz_in_row = [&](const IndexType i, const IndexType j) {
        int num_neighbors = 0;
        for (IndexType d_i : {-1, 0, 1}) {
            for (IndexType d_j : {-1, 0, 1}) {
                if (d_i != 0 && d_j != 0) {
                    auto neighbor = flat_idx(j + d_j, i + d_i);
                    if (is_valid_idx(neighbor)) {
                        num_neighbors++;
                    }
                }
            }
        }
        return num_neighbors;
    };

    for (IndexType i = 0; i < dp; ++i) {
        for (IndexType j = 0; j < dp; ++j) {
            auto row = flat_idx(j, i);
            auto diag_value = static_cast<ValueType>(nnz_in_row(i, j));
            for (IndexType d_i : {-1, 0, 1}) {
                for (IndexType d_j : {-1, 0, 1}) {
                    if (!restricted || ((d_i == 0 && d_j == 0))) {
                        auto col = flat_idx(j + d_j, i + d_i);
                        if (is_valid_idx(col)) {
                            if (col != row) {
                                A_data.nonzeros.emplace_back(
                                    row, col, -gko::one<ValueType>());
                            } else {
                                A_data.nonzeros.emplace_back(row, col,
                                                             diag_value);
                            }
                        }
                    }
                }
            }
        }
    }

    return A_data;
}


/**
 * Generates matrix data for a 3D stencil matrix. If restricted is set to true,
 * creates a 7-pt stencil, if it is false creates a 27-pt stencil. If
 * strong_scaling is set to true, creates the same problem size independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_3d_stencil(
    gko::mpi::communicator comm, const gko::size_type target_local_size,
    bool restricted)
{
    std::array<int, 3> dims{};
    std::array<int, 3> periods{};
    MPI_Dims_create(comm.size(), dims.size(), dims.data());

    MPI_Comm cart_comm;
    MPI_Cart_create(comm.get(), dims.size(), dims.data(), periods.data(), 0,
                    &cart_comm);

    std::array<int, 3> coords{};
    MPI_Cart_coords(cart_comm, comm.rank(), coords.size(), coords.data());

    const auto dp =
        static_cast<IndexType>(closest_nth_root(target_local_size, 3));
    const auto local_size = static_cast<gko::size_type>(dp * dp * dp);
    const auto global_size = local_size * comm.size();
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{global_size, global_size});

    auto global_offset = [&](const int cx, const int cy, const int cz) {
        return cx * static_cast<int>(local_size) +
               cy * static_cast<int>(local_size) * dims[0] +
               cz * static_cast<int>(local_size * local_size) * dims[0] *
                   dims[1];
    };

    auto target_coords = [&](const IndexType i, const int coord) {
        return 0 <= i && i <= dp - 1 ? coord : i < 0 ? coord - 1 : coord + 1;
    };

    auto target_local_idx = [&](const IndexType i) {
        return 0 <= i && i <= dp - 1 ? i : i < 0 ? dp + i : dp - i;
    };

    auto flat_idx = [&](const IndexType ix, const IndexType iy,
                        const IndexType iz) {
        return global_offset(target_coords(ix, coords[0]),
                             target_coords(iy, coords[1]),
                             target_coords(iz, coords[2])) +
               target_local_idx(ix) + target_local_idx(iy) * dp +
               target_local_idx(iz) * dp * dp;
    };

    for (IndexType i = 0; i < dp; ++i) {
        for (IndexType j = 0; j < dp; ++j) {
            for (IndexType k = 0; k < dp; ++k) {
                auto row = flat_idx(k, j, i);
                for (IndexType d_i : {-1, 0, 1}) {
                    for (IndexType d_j : {-1, 0, 1}) {
                        for (IndexType d_k : {-1, 0, 1}) {
                            if (!restricted || ((d_i == 0 && d_j == 0) ||
                                                (d_i == 0 && d_k == 0) ||
                                                (d_j == 0 && d_k == 0))) {
                                auto col = flat_idx(k + d_k, j + d_j, i + d_i);
                                if (col >= 0 && col < global_size) {
                                    A_data.nonzeros.emplace_back(
                                        row, col, gko::one<ValueType>());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return A_data;
}


/**
 * Generates matrix data for a 2D stencil matrix. If restricted is set to true,
 * creates a 5-pt stencil, if it is false creates a 9-pt stencil. If
 * strong_scaling is set to true, creates the same problemsize independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_stencil_with_optimal_comm(
    gko::mpi::communicator comm, const IndexType target_local_size,
    bool restricted)
{
    const auto dp =
        static_cast<IndexType>(closest_nth_root(target_local_size, 2));
    const auto mat_size = dp * dp * comm.size();
    const auto rows_per_rank = gko::ceildiv(mat_size, comm.size());
    const auto start = rows_per_rank * comm.rank();
    const auto end = gko::min(rows_per_rank * (comm.rank() + 1), mat_size);

    auto A_data =
        gko::matrix_data<ValueType, IndexType>(gko::dim<2>{mat_size, mat_size});

    for (IndexType row = start; row < end; row++) {
        auto i = row / dp;
        auto j = row % dp;
        for (IndexType d_i = -1; d_i <= 1; d_i++) {
            for (IndexType d_j = -1; d_j <= 1; d_j++) {
                if (!restricted || (d_i == 0 || d_j == 0)) {
                    auto col = j + d_j + (i + d_i) * dp;
                    if (col >= 0 && col < mat_size) {
                        A_data.nonzeros.emplace_back(row, col,
                                                     gko::one<ValueType>());
                    }
                }
            }
        }
    }

    return A_data;
}


/**
 * Generates matrix data for a 3D stencil matrix. If restricted is set to true,
 * creates a 7-pt stencil, if it is false creates a 27-pt stencil. If
 * strong_scaling is set to true, creates the same problemsize independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_3d_stencil_with_optimal_comm(
    gko::mpi::communicator comm, const IndexType target_local_size,
    bool restricted)
{
    const auto dp =
        static_cast<IndexType>(closest_nth_root(target_local_size, 3));
    const auto mat_size = dp * dp * dp * comm.size();
    const auto rows_per_rank = gko::ceildiv(mat_size, comm.size());
    const auto start = rows_per_rank * comm.rank();
    const auto end = gko::min(rows_per_rank * (comm.rank() + 1), mat_size);

    auto A_data =
        gko::matrix_data<ValueType, IndexType>(gko::dim<2>{mat_size, mat_size});

    for (IndexType row = start; row < end; row++) {
        auto i = row / (dp * dp);
        auto j = (row % (dp * dp)) / dp;
        auto k = row % dp;
        for (IndexType d_i = -1; d_i <= 1; d_i++) {
            for (IndexType d_j = -1; d_j <= 1; d_j++) {
                for (IndexType d_k = -1; d_k <= 1; d_k++) {
                    if (!restricted ||
                        ((d_i == 0 && d_j == 0) || (d_i == 0 && d_k == 0) ||
                         (d_j == 0 && d_k == 0))) {
                        auto col =
                            k + d_k + (j + d_j) * dp + (i + d_i) * dp * dp;
                        if (col >= 0 && col < mat_size) {
                            A_data.nonzeros.emplace_back(row, col,
                                                         gko::one<ValueType>());
                        }
                    }
                }
            }
        }
    }

    return A_data;
}


template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_stencil(
    std::string stencil_name, gko::mpi::communicator comm,
    const gko::size_type target_local_size, bool optimal_comm)
{
    if (optimal_comm) {
        if (stencil_name == "5pt") {
            return generate_2d_stencil_with_optimal_comm<ValueType, IndexType>(
                std::move(comm), target_local_size, false);
        } else if (stencil_name == "9pt") {
            return generate_2d_stencil_with_optimal_comm<ValueType, IndexType>(
                std::move(comm), target_local_size, true);
        } else if (stencil_name == "7pt") {
            return generate_3d_stencil_with_optimal_comm<ValueType, IndexType>(
                std::move(comm), target_local_size, false);
        } else if (stencil_name == "27pt") {
            return generate_3d_stencil_with_optimal_comm<ValueType, IndexType>(
                std::move(comm), target_local_size, true);
        } else {
            throw std::runtime_error("Stencil " + stencil_name +
                                     " not implemented");
        }
    } else {
        if (stencil_name == "5pt") {
            return generate_2d_stencil<ValueType, IndexType>(
                std::move(comm), target_local_size, false);
        } else if (stencil_name == "9pt") {
            return generate_2d_stencil<ValueType, IndexType>(
                std::move(comm), target_local_size, true);
        } else if (stencil_name == "7pt") {
            return generate_3d_stencil<ValueType, IndexType>(
                std::move(comm), target_local_size, false);
        } else if (stencil_name == "27pt") {
            return generate_3d_stencil<ValueType, IndexType>(
                std::move(comm), target_local_size, true);
        } else {
            throw std::runtime_error("Stencil " + stencil_name +
                                     " not implemented");
        }
    }
}
#endif  // GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP
