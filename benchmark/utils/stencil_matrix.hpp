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


#include <ginkgo/core/base/matrix_data.hpp>
#if GINKGO_BUILD_MPI
#include <ginkgo/core/base/mpi.hpp>
#endif


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


template <typename IndexType>
bool is_in_box(const IndexType i, const IndexType bound)
{
    return 0 <= i && i < bound;
}


/**
 * Generates matrix data for a 2D stencil matrix. If restricted is set to true,
 * creates a 5-pt stencil, if it is false creates a 9-pt stencil.
 *
 * If `dim != [1 1]` then the matrix data is a subset of a larger matrix.
 * The total matrix is a discretization of `[0, dims[0]] x [0, dims[1]]`, and
 * each box is square. The position of the box defines the subset of the matrix.
 * The degrees of freedom are ordered box-wise and the boxes themselves are
 * ordered lexicographical. This means that the indices are with respect to the
 * larger matrix, i.e. they might not start with 0.
 *
 * @param dims  The number of boxes in each dimension.
 * @param positions  The position of this box with respect to each dimension.
 * @param target_local_size  The desired size of the boxes. The actual size can
 *                           deviate from this to accommodate the square size of
 *                           the boxes.
 * @param restricted  If true, a 5-pt stencil is used, else a 9-pt stencil.
 *
 * @return  matrix data of a box using either 5-pt or 9-pt stencil.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_stencil_box(
    std::array<int, 2> dims, std::array<int, 2> positions,
    const gko::size_type target_local_size, bool restricted)
{
    auto num_boxes =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});

    const auto discretization_points =
        static_cast<IndexType>(closest_nth_root(target_local_size, 2));
    const auto local_size = static_cast<gko::size_type>(discretization_points *
                                                        discretization_points);
    const auto global_size = local_size * num_boxes;
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)});

    auto global_offset = [&](const int position_x, const int position_y) {
        return static_cast<int>(local_size) * position_x +
               static_cast<int>(local_size) * dims[0] * position_y;
    };

    auto target_position = [&](const IndexType i, const int position) {
        return is_in_box(i, discretization_points)
                   ? position
                   : (i < 0 ? position - 1 : position + 1);
    };

    auto target_local_idx = [&](const IndexType i) {
        return is_in_box(i, discretization_points)
                   ? i
                   : (i < 0 ? discretization_points + i
                            : discretization_points - i);
    };

    auto flat_idx = [&](const IndexType ix, const IndexType iy) {
        auto tpx = target_position(ix, positions[0]);
        auto tpy = target_position(iy, positions[1]);
        if (is_in_box(tpx, dims[0]) && is_in_box(tpy, dims[1])) {
            return global_offset(tpx, tpy) + target_local_idx(ix) +
                   target_local_idx(iy) * discretization_points;
        } else {
            return static_cast<IndexType>(-1);
        }
    };

    auto is_valid_neighbor = [&](const IndexType d_i, const IndexType d_j) {
        return !restricted || d_i == 0 || d_j == 0;
    };

    auto nnz_in_row = [&]() {
        int num_neighbors = 0;
        for (IndexType d_i : {-1, 0, 1}) {
            for (IndexType d_j : {-1, 0, 1}) {
                if (is_valid_neighbor(d_i, d_j)) {
                    num_neighbors++;
                }
            }
        }
        return num_neighbors;
    };
    const auto diag_value = static_cast<ValueType>(nnz_in_row() - 1);

    for (IndexType i = 0; i < discretization_points; ++i) {
        for (IndexType j = 0; j < discretization_points; ++j) {
            auto row = flat_idx(j, i);
            for (IndexType d_i : {-1, 0, 1}) {
                for (IndexType d_j : {-1, 0, 1}) {
                    if (is_valid_neighbor(d_i, d_j)) {
                        auto col = flat_idx(j + d_j, i + d_i);
                        if (is_in_box(col,
                                      static_cast<IndexType>(global_size))) {
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
 * creates a 7-pt stencil, if it is false creates a 27-pt stencil.
 *
 * If `dim != [1 1 1]` then the matrix data is a subset of a larger matrix.
 * The total matrix is a discretization of `[0, dims[0]] x [0, dims[1]] x [0,
 * dims[2]]`, and each box is a cube. The position of the box defines the subset
 * of the matrix. The degrees of freedom are ordered box-wise and the boxes
 * themselves are ordered lexicographical. This means that the indices are with
 * respect to the larger matrix, i.e. they might not start with 0.
 *
 * @param dims  The number of boxes in each dimension.
 * @param positions  The position of this box with respect to each dimension.
 * @param target_local_size  The desired size of the boxes. The actual size can
 *                           deviate from this to accommodate the uniform size
 *                           of the boxes.
 * @param restricted  If true, a 7-pt stencil is used, else a 27-pt stencil.
 *
 * @return  matrix data of a box using either 7-pt or 27-pt stencil.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_3d_stencil_box(
    std::array<int, 3> dims, std::array<int, 3> positions,
    const gko::size_type target_local_size, bool restricted)
{
    auto num_boxes =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});

    const auto discretization_points =
        static_cast<IndexType>(closest_nth_root(target_local_size, 3));
    const auto local_size = static_cast<gko::size_type>(
        discretization_points * discretization_points * discretization_points);
    const auto global_size = local_size * num_boxes;
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)});

    auto global_offset = [&](const int position_x, const int position_y,
                             const int position_z) {
        return position_x * static_cast<int>(local_size) +
               position_y * static_cast<int>(local_size) * dims[0] +
               position_z * static_cast<int>(local_size) * dims[0] * dims[1];
    };

    auto target_position = [&](const IndexType i, const int position) {
        return is_in_box(i, discretization_points)
                   ? position
                   : (i < 0 ? position - 1 : position + 1);
    };

    auto target_local_idx = [&](const IndexType i) {
        return is_in_box(i, discretization_points)
                   ? i
                   : (i < 0 ? discretization_points + i
                            : discretization_points - i);
    };

    auto flat_idx = [&](const IndexType ix, const IndexType iy,
                        const IndexType iz) {
        auto tpx = target_position(ix, positions[0]);
        auto tpy = target_position(iy, positions[1]);
        auto tpz = target_position(iz, positions[2]);
        if (is_in_box(tpx, dims[0]) && is_in_box(tpy, dims[1]) &&
            is_in_box(tpz, dims[2])) {
            return global_offset(tpx, tpy, tpz) + target_local_idx(ix) +
                   target_local_idx(iy) * discretization_points +
                   target_local_idx(iz) * discretization_points *
                       discretization_points;
        } else {
            return static_cast<IndexType>(-1);
        }
    };

    auto is_valid_neighbor = [&](const IndexType d_i, const IndexType d_j,
                                 const IndexType d_k) {
        return !restricted || ((d_i == 0) + (d_j == 0) + (d_k == 0) >= 2);
    };

    auto nnz_in_row = [&]() {
        int num_neighbors = 0;
        for (IndexType d_i : {-1, 0, 1}) {
            for (IndexType d_j : {-1, 0, 1}) {
                for (IndexType d_k : {-1, 0, 1}) {
                    if (is_valid_neighbor(d_i, d_j, d_k)) {
                        num_neighbors++;
                    }
                }
            }
        }
        return num_neighbors;
    };
    const auto diag_value = static_cast<ValueType>(nnz_in_row() - 1);

    for (IndexType i = 0; i < discretization_points; ++i) {
        for (IndexType j = 0; j < discretization_points; ++j) {
            for (IndexType k = 0; k < discretization_points; ++k) {
                auto row = flat_idx(k, j, i);
                for (IndexType d_i : {-1, 0, 1}) {
                    for (IndexType d_j : {-1, 0, 1}) {
                        for (IndexType d_k : {-1, 0, 1}) {
                            if (is_valid_neighbor(d_i, d_j, d_k)) {
                                auto col = flat_idx(k + d_k, j + d_j, i + d_i);
                                if (is_in_box(col, static_cast<IndexType>(
                                                       global_size))) {
                                    if (col != row) {
                                        A_data.nonzeros.emplace_back(
                                            row, col, -gko::one<ValueType>());
                                    } else {
                                        A_data.nonzeros.emplace_back(
                                            row, col, diag_value);
                                    }
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
 * Generates matrix data for the requested stencil.
 *
 * @see generate_2d_stencil_box, generate_3d_stencil_box
 *
 * @param stencil_name  The name of the stencil.
 * @param target_local_size  The desired size of the matrix. The actual size can
 *                           deviate from this to accommodate the uniform size
 *                           of the discretization.
 * @return  matrix data using the requested stencil.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_stencil(
    std::string stencil_name, const gko::size_type target_local_size)
{
    if (stencil_name == "5pt") {
        return generate_2d_stencil_box<ValueType, IndexType>(
            {1, 1}, {0, 0}, target_local_size, true);
    } else if (stencil_name == "9pt") {
        return generate_2d_stencil_box<ValueType, IndexType>(
            {1, 1}, {0, 0}, target_local_size, false);
    } else if (stencil_name == "7pt") {
        return generate_3d_stencil_box<ValueType, IndexType>(
            {1, 1, 1}, {0, 0, 0}, target_local_size, true);
    } else if (stencil_name == "27pt") {
        return generate_3d_stencil_box<ValueType, IndexType>(
            {1, 1, 1}, {0, 0, 0}, target_local_size, false);
    } else {
        throw std::runtime_error("Stencil " + stencil_name +
                                 " not implemented");
    }
}


#if GINKGO_BUILD_MPI


/**
 * Generates matrix data for a given 2D stencil, where the position of this
 * block is given by it's MPI rank.
 *
 * @see generate_2d_stencil_box
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_stencil(
    gko::experimental::mpi::communicator comm,
    const gko::size_type target_local_size, bool restricted)
{
    std::array<int, 2> dims{};
    MPI_Dims_create(comm.size(), dims.size(), dims.data());

    std::array<int, 2> coords{};
    coords[0] = comm.rank() % dims[0];
    coords[1] = comm.rank() / dims[0];

    return generate_2d_stencil_box<ValueType, IndexType>(
        dims, coords, target_local_size, restricted);
}


/**
 * Generates matrix data for a given 23 stencil, where the position of this
 * block is given by it's MPI rank.
 *
 * @see generate_3d_stencil_box
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_3d_stencil(
    gko::experimental::mpi::communicator comm,
    const gko::size_type target_local_size, bool restricted)
{
    std::array<int, 3> dims{};
    MPI_Dims_create(comm.size(), dims.size(), dims.data());

    std::array<int, 3> coords{};
    coords[0] = comm.rank() % dims[0];
    coords[1] = (comm.rank() / dims[0]) % dims[1];
    coords[2] = comm.rank() / (dims[0] * dims[1]);

    return generate_3d_stencil_box<ValueType, IndexType>(
        dims, coords, target_local_size, restricted);
}


/**
 * Generates matrix data for a 2D stencil matrix. If restricted is set to true,
 * creates a 5-pt stencil, if it is false creates a 9-pt stencil.
 *
 * The degrees of freedom are ordered such that each MPI rank has at most two
 * neighbors.
 *
 * @see generate_2d_stencil_box
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_stencil_with_optimal_comm(
    gko::experimental::mpi::communicator comm,
    const IndexType target_local_size, bool restricted)
{
    const auto discretization_points =
        static_cast<IndexType>(closest_nth_root(target_local_size, 2));
    const auto mat_size =
        discretization_points * discretization_points * comm.size();
    const auto rows_per_rank = gko::ceildiv(mat_size, comm.size());
    const auto start = rows_per_rank * comm.rank();
    const auto end = gko::min(rows_per_rank * (comm.rank() + 1), mat_size);

    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{static_cast<gko::size_type>(mat_size),
                    static_cast<gko::size_type>(mat_size)});

    auto is_valid_neighbor = [&](const IndexType d_i, const IndexType d_j) {
        return !restricted || ((d_i == 0 || d_j == 0));
    };

    auto nnz_in_row = [&]() {
        int num_neighbors = 0;
        for (IndexType d_i : {-1, 0, 1}) {
            for (IndexType d_j : {-1, 0, 1}) {
                if (is_valid_neighbor(d_i, d_j)) {
                    num_neighbors++;
                }
            }
        }
        return num_neighbors;
    };
    const auto diag_value = static_cast<ValueType>(nnz_in_row() - 1);

    for (IndexType row = start; row < end; row++) {
        auto i = row / discretization_points;
        auto j = row % discretization_points;
        for (IndexType d_i = -1; d_i <= 1; d_i++) {
            for (IndexType d_j = -1; d_j <= 1; d_j++) {
                if (is_valid_neighbor(d_i, d_j)) {
                    auto col = j + d_j + (i + d_i) * discretization_points;
                    if (is_in_box(col, mat_size)) {
                        if (col != row) {
                            A_data.nonzeros.emplace_back(
                                row, col, -gko::one<ValueType>());
                        } else {
                            A_data.nonzeros.emplace_back(row, col, diag_value);
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
 * creates a 7-pt stencil, if it is false creates a 27-pt stencil.
 *
 * The degrees of freedom are ordered such that each MPI rank has at most two
 * neighbors.
 *
 * @see generate_3d_stencil_box
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_3d_stencil_with_optimal_comm(
    gko::experimental::mpi::communicator comm,
    const IndexType target_local_size, bool restricted)
{
    const auto discretization_points =
        static_cast<IndexType>(closest_nth_root(target_local_size, 3));
    const auto mat_size = discretization_points * discretization_points *
                          discretization_points * comm.size();
    const auto rows_per_rank = gko::ceildiv(mat_size, comm.size());
    const auto start = rows_per_rank * comm.rank();
    const auto end = gko::min(rows_per_rank * (comm.rank() + 1), mat_size);

    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{static_cast<gko::size_type>(mat_size),
                    static_cast<gko::size_type>(mat_size)});

    auto is_valid_neighbor = [&](const IndexType d_i, const IndexType d_j,
                                 const IndexType d_k) {
        return !restricted || ((d_i == 0) + (d_j == 0) + (d_k == 0) >= 2);
    };

    auto nnz_in_row = [&]() {
        int num_neighbors = 0;
        for (IndexType d_i : {-1, 0, 1}) {
            for (IndexType d_j : {-1, 0, 1}) {
                for (IndexType d_k : {-1, 0, 1}) {
                    if (is_valid_neighbor(d_i, d_j, d_k)) {
                        num_neighbors++;
                    }
                }
            }
        }
        return num_neighbors;
    };
    const auto diag_value = static_cast<ValueType>(nnz_in_row() - 1);

    for (IndexType row = start; row < end; row++) {
        auto i = row / (discretization_points * discretization_points);
        auto j = (row % (discretization_points * discretization_points)) /
                 discretization_points;
        auto k = row % discretization_points;
        for (IndexType d_i = -1; d_i <= 1; d_i++) {
            for (IndexType d_j = -1; d_j <= 1; d_j++) {
                for (IndexType d_k = -1; d_k <= 1; d_k++) {
                    if (is_valid_neighbor(d_i, d_j, d_k)) {
                        auto col = k + d_k + (j + d_j) * discretization_points +
                                   (i + d_i) * discretization_points *
                                       discretization_points;
                        if (is_in_box(col, mat_size)) {
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
 * Generates matrix data for the requested stencil.
 *
 * @copydoc  generate_stencil(const gko::size_type, bool)
 *
 * @param comm  The MPI communicator to determine the rank.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_stencil(
    std::string stencil_name, gko::experimental::mpi::communicator comm,
    const gko::size_type target_local_size, bool optimal_comm)
{
    if (optimal_comm) {
        if (stencil_name == "5pt") {
            return generate_2d_stencil_with_optimal_comm<ValueType, IndexType>(
                std::move(comm), target_local_size, true);
        } else if (stencil_name == "9pt") {
            return generate_2d_stencil_with_optimal_comm<ValueType, IndexType>(
                std::move(comm), target_local_size, false);
        } else if (stencil_name == "7pt") {
            return generate_3d_stencil_with_optimal_comm<ValueType, IndexType>(
                std::move(comm), target_local_size, true);
        } else if (stencil_name == "27pt") {
            return generate_3d_stencil_with_optimal_comm<ValueType, IndexType>(
                std::move(comm), target_local_size, false);
        } else {
            throw std::runtime_error("Stencil " + stencil_name +
                                     " not implemented");
        }
    } else {
        if (stencil_name == "5pt") {
            return generate_2d_stencil<ValueType, IndexType>(
                std::move(comm), target_local_size, true);
        } else if (stencil_name == "9pt") {
            return generate_2d_stencil<ValueType, IndexType>(
                std::move(comm), target_local_size, false);
        } else if (stencil_name == "7pt") {
            return generate_3d_stencil<ValueType, IndexType>(
                std::move(comm), target_local_size, true);
        } else if (stencil_name == "27pt") {
            return generate_3d_stencil<ValueType, IndexType>(
                std::move(comm), target_local_size, false);
        } else {
            throw std::runtime_error("Stencil " + stencil_name +
                                     " not implemented");
        }
    }
}


#endif
#endif  // GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP
