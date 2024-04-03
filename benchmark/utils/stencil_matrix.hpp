// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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

/**
 * Checks if an index is within [0, bound).
 *
 * @return True if i in [0, bound).
 */
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

    /**
     * This computes the offsets in the global indices for a box at (position_y,
     * position_x).
     */
    auto global_offset = [&](const IndexType position_y,
                             const IndexType position_x) {
        return static_cast<IndexType>(local_size) * position_x +
               static_cast<IndexType>(local_size) * dims[0] * position_y;
    };

    /**
     * This computes a single dimension of the target box position
     * for a given index. The target box is the box that owns the given index.
     * If the index is within the local indices [0, discretization_points) this
     * returns the current position, otherwise it is shifted by +-1.
     */
    auto target_position = [&](const IndexType i, const int position) {
        return is_in_box(i, discretization_points)
                   ? position
                   : (i < 0 ? position - 1 : position + 1);
    };

    /**
     * This computes a single dimension of target local index for a given index.
     * The target local index is the local index within the index set of the box
     * that owns the index. If the index is within the local indices [0,
     * discretization_points), this returns the index unchanged, otherwise it is
     * projected into the index set of the owning, adjacent box.
     */
    auto target_local_idx = [&](const IndexType i) {
        return is_in_box(i, discretization_points)
                   ? i
                   : (i < 0 ? discretization_points + i
                            : discretization_points - i);
    };

    /**
     * For a two dimensional pair of local indices (iy, ix), this computes the
     * corresponding global one dimensional index.
     * If any target positions of the owning box is not inside [0, dims[0]] x
     * [0, dims[1]] then the invalid index -1 is returned.
     */
    auto flat_idx = [&](const IndexType iy, const IndexType ix) {
        auto tpx = target_position(ix, positions[0]);
        auto tpy = target_position(iy, positions[1]);
        if (is_in_box(tpx, dims[0]) && is_in_box(tpy, dims[1])) {
            return global_offset(tpy, tpx) + target_local_idx(ix) +
                   target_local_idx(iy) * discretization_points;
        } else {
            return static_cast<IndexType>(-1);
        }
    };

    /**
     * This computes if the given local-index-offsets are valid neighbors for
     * the current stencil, regardless of any global indices.
     */
    auto is_valid_neighbor = [&](const IndexType dy, const IndexType dx) {
        return !restricted || dx == 0 || dy == 0;
    };

    /**
     * This computes the maximal number of non-zeros per row for the current
     * stencil.
     */
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

    A_data.nonzeros.reserve(nnz_in_row() * local_size);

    for (IndexType iy = 0; iy < discretization_points; ++iy) {
        for (IndexType ix = 0; ix < discretization_points; ++ix) {
            auto row = flat_idx(iy, ix);
            for (IndexType dy : {-1, 0, 1}) {
                for (IndexType dx : {-1, 0, 1}) {
                    if (is_valid_neighbor(dy, dx)) {
                        auto col = flat_idx(iy + dy, ix + dx);
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

    /**
     * This computes the offsets in the global indices for a box at (position_z,
     * position_y, position_x).
     */
    auto global_offset = [&](const IndexType position_z,
                             const IndexType position_y,
                             const IndexType position_x) {
        return position_x * static_cast<IndexType>(local_size) +
               position_y * static_cast<IndexType>(local_size) * dims[0] +
               position_z * static_cast<IndexType>(local_size) * dims[0] *
                   dims[1];
    };

    /**
     * This computes a single dimension of the target box position
     * for a given index. The target box is the box that owns the given index.
     * If the index is within the local indices [0, discretization_points) this
     * returns the current position, otherwise it is shifted by +-1.
     */
    auto target_position = [&](const IndexType i, const int position) {
        return is_in_box(i, discretization_points)
                   ? position
                   : (i < 0 ? position - 1 : position + 1);
    };

    /**
     * This computes a single dimension of target local index for a given index.
     * The target local index is the local index within the index set of the box
     * that owns the index. If the index is within the local indices [0,
     * discretization_points), this returns the index unchanged, otherwise it is
     * projected into the index set of the owning, adjacent box.
     */
    auto target_local_idx = [&](const IndexType i) {
        return is_in_box(i, discretization_points)
                   ? i
                   : (i < 0 ? discretization_points + i
                            : discretization_points - i);
    };

    /**
     * For a three dimensional tuple of local indices (iz, iy, ix), this
     * computes the corresponding global one dimensional index. If any target
     * positions of the owning box is not inside [0, dims[0]] x [0, dims[1]] x
     * [0, dims[2]] then the invalid index -1 is returned.
     */
    auto flat_idx = [&](const IndexType iz, const IndexType iy,
                        const IndexType ix) {
        auto tpx = target_position(ix, positions[0]);
        auto tpy = target_position(iy, positions[1]);
        auto tpz = target_position(iz, positions[2]);
        if (is_in_box(tpx, dims[0]) && is_in_box(tpy, dims[1]) &&
            is_in_box(tpz, dims[2])) {
            return global_offset(tpz, tpy, tpx) + target_local_idx(ix) +
                   target_local_idx(iy) * discretization_points +
                   target_local_idx(iz) * discretization_points *
                       discretization_points;
        } else {
            return static_cast<IndexType>(-1);
        }
    };

    /**
     * This computes if the given local-index-offsets are valid neighbors for
     * the current stencil, regardless of any global indices.
     */
    auto is_valid_neighbor = [&](const IndexType dz, const IndexType dy,
                                 const IndexType dx) {
        return !restricted || ((dz == 0) + (dy == 0) + (dx == 0) >= 2);
    };

    /**
     * This computes the maximal number of non-zeros per row for the current
     * stencil.
     */
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

    A_data.nonzeros.reserve(nnz_in_row() * local_size);

    for (IndexType iz = 0; iz < discretization_points; ++iz) {
        for (IndexType iy = 0; iy < discretization_points; ++iy) {
            for (IndexType ix = 0; ix < discretization_points; ++ix) {
                auto row = flat_idx(iz, iy, ix);
                for (IndexType dz : {-1, 0, 1}) {
                    for (IndexType dy : {-1, 0, 1}) {
                        for (IndexType dx : {-1, 0, 1}) {
                            if (is_valid_neighbor(dz, dy, dx)) {
                                auto col = flat_idx(iz + dz, iy + dy, ix + dx);
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


gko::size_type convert_to_arrow_head_size(gko::size_type orig_size,
                                          gko::size_type problem_dim,
                                          const std::string& size)
{
    if (size == "dim-0") {
        return 1;
    }

    const auto n = static_cast<gko::size_type>(
        closest_nth_root(orig_size, static_cast<int>(problem_dim)));

    if (size == "dim-1") {
        return n;
    }
    if (size == "dim-2") {
        return n * n;
    }
    if (size == "dim-3") {
        return n * n * n;
    }

    return static_cast<gko::size_type>(std::stoll(size));
}


template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> add_arrow(
    const gko::matrix_data<ValueType, IndexType>& orig,
    const std::string& arrow_head_size, gko::size_type problem_dim)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(orig.size);
    const auto size = orig.size[0];

    const auto num_arrow_dofs =
        convert_to_arrow_head_size(size, problem_dim, arrow_head_size);
    const gko::size_type num_connection_per_arrow_dof = size / num_arrow_dofs;

    // diagonal value will be 10
    // offdiagonal values are -(10 / N) to keep pos. definiteness
    const auto diag_value = static_cast<ValueType>(10);
    const auto offidag_value =
        -diag_value / static_cast<ValueType>(num_connection_per_arrow_dof);

    gko::matrix_assembly_data<ValueType, IndexType> data{
        {size + num_arrow_dofs, size + num_arrow_dofs}};

    for (const auto& e : orig.nonzeros) {
        data.set_value(e.row, e.column, e.value);
    }

    for (IndexType i = size; i < static_cast<IndexType>(size + num_arrow_dofs);
         ++i) {
        for (IndexType j = size;
             j < static_cast<IndexType>(size + num_arrow_dofs); ++j) {
            if (i == j) {
                data.set_value(i, j, diag_value);
            } else {
                data.set_value(i, j, offidag_value);
            }
        }
    }

    for (gko::size_type arrow_dof = 0; arrow_dof < num_arrow_dofs;
         ++arrow_dof) {
        for (gko::size_type connection = 0;
             connection < num_connection_per_arrow_dof; ++connection) {
            data.set_value(
                static_cast<IndexType>(size + arrow_dof),
                static_cast<IndexType>(connection * num_arrow_dofs + arrow_dof),
                offidag_value);
            data.set_value(
                static_cast<IndexType>(connection * num_arrow_dofs + arrow_dof),
                static_cast<IndexType>(size + arrow_dof), offidag_value);
        }
    }
    return data.get_ordered_data();
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
gko::matrix_data<ValueType, IndexType> generate_stencil(const json& config)
{
    auto stencil_name = config.at("stencil").get<std::string>();
    auto target_local_size = config.at("size").get<gko::int64>();

    gko::size_type problem_dim;
    gko::matrix_data<ValueType, IndexType> data;
    if (stencil_name == "5pt") {
        problem_dim = 2;
        data = generate_2d_stencil_box<ValueType, IndexType>(
            {1, 1}, {0, 0}, target_local_size, true);
    } else if (stencil_name == "9pt") {
        problem_dim = 2;
        data = generate_2d_stencil_box<ValueType, IndexType>(
            {1, 1}, {0, 0}, target_local_size, false);
    } else if (stencil_name == "7pt") {
        problem_dim = 3;
        data = generate_3d_stencil_box<ValueType, IndexType>(
            {1, 1, 1}, {0, 0, 0}, target_local_size, true);
    } else if (stencil_name == "27pt") {
        problem_dim = 3;
        data = generate_3d_stencil_box<ValueType, IndexType>(
            {1, 1, 1}, {0, 0, 0}, target_local_size, false);
    } else {
        throw std::runtime_error("Stencil " + stencil_name +
                                 " not implemented");
    }

    if (config.contains("extra_structure")) {
        auto type = config["extra_structure"].value("type", "none");
        if (type == "arrow") {
            auto size = config["extra_structure"].value("size", "1");
            data = add_arrow(data, size, problem_dim);
        } else if (type != "none") {
            throw std::runtime_error("Requested extra structure <" + type +
                                     "> not known.");
        }
    }

    return data;
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
    const gko::size_type target_local_size, bool restricted, bool optimal_comm)
{
    if (optimal_comm) {
        return generate_2d_stencil_box<ValueType, IndexType>(
            {comm.size(), 1}, {comm.rank(), 0}, target_local_size, restricted);
    } else {
        std::array<int, 2> dims{};
        MPI_Dims_create(comm.size(), dims.size(), dims.data());

        std::array<int, 2> coords{};
        coords[0] = comm.rank() % dims[0];
        coords[1] = comm.rank() / dims[0];

        return generate_2d_stencil_box<ValueType, IndexType>(
            dims, coords, target_local_size, restricted);
    }
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
    const gko::size_type target_local_size, bool restricted, bool optimal_comm)
{
    if (optimal_comm) {
        return generate_3d_stencil_box<ValueType, IndexType>(
            {comm.size(), 1, 1}, {comm.rank(), 0, 0}, target_local_size,
            restricted);
    } else {
        std::array<int, 3> dims{};

        MPI_Dims_create(comm.size(), dims.size(), dims.data());

        std::array<int, 3> coords{};
        coords[0] = comm.rank() % dims[0];
        coords[1] = (comm.rank() / dims[0]) % dims[1];
        coords[2] = comm.rank() / (dims[0] * dims[1]);

        return generate_3d_stencil_box<ValueType, IndexType>(
            dims, coords, target_local_size, restricted);
    }
}


/**
 * Generates matrix data for the requested stencil.
 *
 * @copydoc  generate_stencil(const gko::size_type, bool)
 *
 * @param comm  The MPI communicator to determine the rank.
 * @param optimal_comm  If true, a  1D domain decomposition is used which leads
 *                      to each processor having at most two neighbors. This
 *                      also changes the domain shape to an elongated channel.
 *                      If false, a mostly uniform 2D or 3D decomposition is
 *                      used, and the domain shape is mostly cubic.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_stencil(
    std::string stencil_name, gko::experimental::mpi::communicator comm,
    const gko::size_type target_local_size, bool optimal_comm)
{
    if (stencil_name == "5pt") {
        return generate_2d_stencil<ValueType, IndexType>(
            std::move(comm), target_local_size, true, optimal_comm);
    } else if (stencil_name == "9pt") {
        return generate_2d_stencil<ValueType, IndexType>(
            std::move(comm), target_local_size, false, optimal_comm);
    } else if (stencil_name == "7pt") {
        return generate_3d_stencil<ValueType, IndexType>(
            std::move(comm), target_local_size, true, optimal_comm);
    } else if (stencil_name == "27pt") {
        return generate_3d_stencil<ValueType, IndexType>(
            std::move(comm), target_local_size, false, optimal_comm);
    } else {
        throw std::runtime_error("Stencil " + stencil_name +
                                 " not implemented");
    }
}


#endif
#endif  // GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP
