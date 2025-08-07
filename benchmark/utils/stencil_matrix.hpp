// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP
#define GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP


#include <ginkgo/core/base/matrix_data.hpp>

#if GINKGO_BUILD_MPI

#include <ginkgo/core/base/mpi.hpp>

#endif

#include <random>

#include <gflags/gflags.h>

DEFINE_bool(standard_stencil, true,
            "If false, stecil offdiag is -1 + rand([0, 0.5))");


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
bool is_in_range(const IndexType i, const IndexType bound)
{
    return 0 <= i && i < bound;
}


/**
 * Generates matrix data for a 2D stencil matrix. If restricted is set to true,
 * creates a 5-pt stencil, if it is false creates a 9-pt stencil.
 *
 * If `dim != [1 1]` then the matrix data is a subset of a larger matrix.
 * The total matrix is a discretization of `[0, 1]^2`, and each subdomain has
 * (roughly) the shape `global_size_1d / dims[0] x global_size_1d / dims[1]`.
 * The position of the subdomain defines the subset of the matrix.
 * The degrees of freedom are ordered subdomain-wise and the subdomains
 * themselves are ordered lexicographical. This means that the indices are with
 * respect to the larger matrix, i.e. they might not start with 0.
 *
 * @param dims  The number of subdomains in each dimension.
 * @param positions  The position of this subdomain with respect to each
 *                   dimension.
 * @param target_local_size  The desired size of the subdomains. The actual size
 *                           can deviate from this to accommodate the square
 *                           size of the global domain.
 * @param restricted  If true, a 5-pt stencil is used, else a 9-pt stencil.
 *
 * @return  pair of (matrix data, local size) of a subdomain using either 5-pt
 *          or 9-pt stencil.
 */
template <typename ValueType, typename IndexType>
std::pair<gko::matrix_data<ValueType, IndexType>, gko::dim<2>>
generate_2d_stencil_subdomain(std::array<int, 2> dims,
                              std::array<int, 2> positions,
                              const gko::size_type target_local_size,
                              bool restricted)
{
    auto num_subdomains =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});

    const auto target_global_size = target_local_size * num_subdomains;
    const auto global_discretization_points =
        static_cast<IndexType>(closest_nth_root(target_global_size, 2));

    // The rounded-down number of local discrectization points per dimension
    // and its rest.
    const std::array<IndexType, 2> discretization_points_min = {
        global_discretization_points / dims[0],
        global_discretization_points / dims[1]};
    const std::array<IndexType, 2> discretization_points_rest = {
        global_discretization_points % dims[0],
        global_discretization_points % dims[1]};

    /**
     * The subdomain size in a single dimension. This is either
     * discretization_points_min[dim], or discretization_points_min[dim]+1.
     * The first R process have the +1 added, such that the sum of the
     * subdomain size over all processes equals to the
     * global_discretization_points.
     */
    auto subdomain_size_1d = [&](const IndexType dim, const IndexType i) {
        assert(0 <= i && i < dims[dim]);
        return discretization_points_min[dim] +
               (i < discretization_points_rest[dim] ? 1 : 0);
    };

    /**
     * The offset of a subdomain in a single dimension. Since the first R
     * processes have a subdomain size of discretization_points_min[dim]+1, the
     * offset adds min(subdomain_id, R) to
     * discretization_points_min[dim]*subdomain_id
     */
    auto subdomain_offset_1d = [&](const IndexType dim, const IndexType i) {
        assert(0 <= i && i < dims[dim]);
        return discretization_points_min[dim] * i +
               std::min(i, discretization_points_rest[dim]);
    };

    const std::array<IndexType, 2> discretization_points = {
        subdomain_size_1d(0, positions[0]), subdomain_size_1d(1, positions[1])};

    const auto local_size = static_cast<gko::size_type>(
        discretization_points[0] * discretization_points[1]);
    const auto global_size = static_cast<gko::size_type>(
        global_discretization_points * global_discretization_points);
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)});

    /**
     * This computes the offsets in the global indices for a subdomain at
     * (position_y, position_x).
     */
    auto subdomain_offset = [&](const IndexType position_y,
                                const IndexType position_x) {
        return global_discretization_points *
                   subdomain_offset_1d(1, position_y) +
               subdomain_size_1d(1, position_y) *
                   subdomain_offset_1d(0, position_x);
    };

    /**
     * This computes a single dimension of the target subdomain position
     * for a given index. The target subdomain is the subdomain that owns the
     * given index. If the index is within the local indices [0,
     * discretization_points) this returns the current position, otherwise it is
     * shifted by +-1.
     */
    auto target_position = [&](const IndexType dim, const IndexType i,
                               const int position) {
        return is_in_range(i, discretization_points[dim])
                   ? position
                   : (i < 0 ? position - 1 : position + 1);
    };

    /**
     * This computes a single dimension of target local index for a given index.
     * The target local index is the local index within the index set of the
     * subdomain that owns the index. If the index is within the local indices
     * [0, discretization_points), this returns the index unchanged, otherwise
     * it is projected into the index set of the owning, adjacent subdomain.
     */
    auto target_local_idx = [&](const IndexType dim, const IndexType pos,
                                const IndexType i) {
        return is_in_range(i, subdomain_size_1d(dim, pos))
                   ? i
                   : (i < 0 ? i + subdomain_size_1d(dim, pos)
                            : i - subdomain_size_1d(dim, positions[dim]));
    };

    /**
     * For a two dimensional pair of local indices (iy, ix), this computes the
     * corresponding global one dimensional index.
     * If any target positions of the owning subdomain is not inside [0,
     * dims[0]] x [0, dims[1]] then the invalid index -1 is returned.
     */
    auto flat_idx = [&](const IndexType iy, const IndexType ix) {
        auto tpx = target_position(0, ix, positions[0]);
        auto tpy = target_position(1, iy, positions[1]);
        if (is_in_range(tpx, dims[0]) && is_in_range(tpy, dims[1])) {
            return subdomain_offset(tpy, tpx) + target_local_idx(0, tpx, ix) +
                   target_local_idx(1, tpy, iy) * subdomain_size_1d(0, tpx);
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

    for (IndexType iy = 0; iy < discretization_points[1]; ++iy) {
        for (IndexType ix = 0; ix < discretization_points[0]; ++ix) {
            auto row = flat_idx(iy, ix);
            for (IndexType dy : {-1, 0, 1}) {
                for (IndexType dx : {-1, 0, 1}) {
                    if (is_valid_neighbor(dy, dx)) {
                        auto col = flat_idx(iy + dy, ix + dx);
                        if (is_in_range(col,
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

    return {A_data, {local_size, local_size}};
}


/**
 * Generates matrix data for a 3D stencil matrix. If restricted is set to true,
 * creates a 7-pt stencil, if it is false creates a 27-pt stencil.
 *
 * If `dim != [1 1 1]` then the matrix data is a subset of a larger matrix.
 * The total matrix is a discretization of `[0, 1]^3`, and each subdomain has
 * (roughly) the shape
 * `global_size_1d / dims[0] x global_size_1d / dims[1] x
 *  global_size_1d / dims[2]`. The position of the subdomain
 * defines the subset of the matrix. The degrees of freedom are ordered
 * subdomain-wise and the subdomains themselves are ordered lexicographical.
 * This means that the indices are with respect to the larger matrix, i.e. they
 * might not start with 0.
 *
 * @param dims  The number of subdomains in each dimension.
 * @param positions  The position of this subdomain with respect to each
 *                   dimension.
 * @param target_local_size  The desired size of the subdomains. The actual size
 *                           can deviate from this to accommodate the uniform
 *                           size of the global domain.
 * @param restricted  If true, a 7-pt stencil is used, else a 27-pt stencil.
 *
 * @return  pair of (matrix data, local size) of a subdomain using either 7-pt
 *          or 27-pt stencil.
 */
template <typename ValueType, typename IndexType>
std::pair<gko::matrix_data<ValueType, IndexType>, gko::dim<2>>
generate_3d_stencil_subdomain(std::array<int, 3> dims,
                              std::array<int, 3> positions,
                              const gko::size_type target_local_size,
                              bool restricted)
{
    auto num_subdomains =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});

    const auto target_global_size = target_local_size * num_subdomains;
    const auto global_discretization_points =
        static_cast<IndexType>(closest_nth_root(target_global_size, 3));

    // The rounded-down number of local discrectization points per dimension
    // and its rest.
    const std::array<IndexType, 3> discretization_points_min = {
        global_discretization_points / dims[0],
        global_discretization_points / dims[1],
        global_discretization_points / dims[2]};
    const std::array<IndexType, 3> discretization_points_rest = {
        global_discretization_points % dims[0],
        global_discretization_points % dims[1],
        global_discretization_points % dims[2]};

    /**
     * The subdomain size in a single dimension. This is either
     * discretization_points_min[dim], or discretization_points_min[dim]+1.
     * The first R process have the +1 added, such that the sum of the
     * subdomain size over all processes equals to the
     * global_discretization_points.
     */
    auto subdomain_size_1d = [&](const IndexType dim, const IndexType i) {
        assert(0 <= i && i < dims[dim]);
        return discretization_points_min[dim] +
               (i < discretization_points_rest[dim] ? 1 : 0);
    };

    /**
     * The offset of a subdomain in a single dimension. Since the first R
     * processes have a subdomain size of discretization_points_min[dim]+1, the
     * offset adds min(subdomain_id, R) to
     * discretization_points_min[dim]*subdomain_id
     */
    auto subdomain_offset_1d = [&](const IndexType dim, const IndexType i) {
        assert(0 <= i && i < dims[dim]);
        return discretization_points_min[dim] * i +
               std::min(i, discretization_points_rest[dim]);
    };

    // The local number of discretization points per dimension
    const std::array<IndexType, 3> discretization_points = {
        subdomain_size_1d(0, positions[0]), subdomain_size_1d(1, positions[1]),
        subdomain_size_1d(2, positions[2])};

    const auto local_size = static_cast<gko::size_type>(
        discretization_points[0] * discretization_points[1] *
        discretization_points[2]);
    const auto global_size = global_discretization_points *
                             global_discretization_points *
                             global_discretization_points;
    auto A_data = gko::matrix_data<ValueType, IndexType>(
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)});

    /**
     * This computes the offsets in the global indices for a subdomain at
     * (position_z, position_y, position_x).
     */
    auto subdomain_offset = [&](const IndexType position_z,
                                const IndexType position_y,
                                const IndexType position_x) {
        return global_discretization_points * global_discretization_points *
                   subdomain_offset_1d(2, position_z) +
               global_discretization_points * subdomain_size_1d(2, position_z) *
                   subdomain_offset_1d(1, position_y) +
               subdomain_size_1d(2, position_z) *
                   subdomain_size_1d(1, position_y) *
                   subdomain_offset_1d(0, position_x);
    };

    /**
     * This computes a single dimension of the target subdomain position
     * for a given index. The target subdomain is the subdomain that owns the
     * given index. If the index is within the local indices [0,
     * discretization_points) this returns the current position, otherwise it is
     * shifted by +-1.
     */
    auto target_position = [&](const IndexType dim, const IndexType i,
                               const int position) {
        return is_in_range(i, discretization_points[dim])
                   ? position
                   : (i < 0 ? position - 1 : position + 1);
    };

    /**
     * This computes a single dimension of target local index for a given index.
     * The target local index is the local index within the index set of the
     * subdomain that owns the index. If the index is within the local indices
     * [0, discretization_points), this returns the index unchanged, otherwise
     * it is projected into the index set of the owning, adjacent subdomain.
     */
    auto target_local_idx = [&](const IndexType dim, const IndexType pos,
                                const IndexType i) {
        return is_in_range(i, subdomain_size_1d(dim, pos))
                   ? i
                   : (i < 0 ? i + subdomain_size_1d(dim, pos)
                            : i - subdomain_size_1d(dim, positions[dim]));
    };

    /**
     * For a three dimensional tuple of local indices (iz, iy, ix), this
     * computes the corresponding global one dimensional index. If any target
     * positions of the owning subdomain is not inside [0, dims[0]] x [0,
     * dims[1]] x [0, dims[2]] then the invalid index -1 is returned.
     */
    auto flat_idx = [&](const IndexType iz, const IndexType iy,
                        const IndexType ix) {
        auto tpx = target_position(0, ix, positions[0]);
        auto tpy = target_position(1, iy, positions[1]);
        auto tpz = target_position(2, iz, positions[2]);
        if (is_in_range(tpx, dims[0]) && is_in_range(tpy, dims[1]) &&
            is_in_range(tpz, dims[2])) {
            return subdomain_offset(tpz, tpy, tpx) +
                   target_local_idx(0, tpx, ix) +
                   target_local_idx(1, tpy, iy) * subdomain_size_1d(0, tpx) +
                   target_local_idx(2, tpz, iz) * subdomain_size_1d(0, tpx) *
                       subdomain_size_1d(1, tpy);
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
    std::mt19937 gen(15);
    std::uniform_real_distribution<> dis(0, 0.5);
    for (IndexType iz = 0; iz < discretization_points[2]; ++iz) {
        for (IndexType iy = 0; iy < discretization_points[1]; ++iy) {
            for (IndexType ix = 0; ix < discretization_points[0]; ++ix) {
                auto row = flat_idx(iz, iy, ix);
                for (IndexType dz : {-1, 0, 1}) {
                    for (IndexType dy : {-1, 0, 1}) {
                        for (IndexType dx : {-1, 0, 1}) {
                            if (is_valid_neighbor(dz, dy, dx)) {
                                auto col = flat_idx(iz + dz, iy + dy, ix + dx);
                                if (is_in_range(col, static_cast<IndexType>(
                                                         global_size))) {
                                    if (col != row) {
                                        A_data.nonzeros.emplace_back(
                                            row, col,
                                            FLAGS_standard_stencil
                                                ? -1
                                                : -1 + dis(gen));
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

    return {A_data, gko::dim<2>{local_size, local_size}};
}


/**
 * Generates matrix data for the requested stencil.
 *
 * @see generate_2d_stencil_subdomain, generate_3d_stencil_subdomain
 *
 * @param stencil_name  The name of the stencil.
 * @param target_local_size  The desired size of the matrix. The actual size can
 *                           deviate from this to accommodate the uniform size
 *                           of the discretization.
 * @return  pair of (matrix data, local size) using the requested stencil.
 */
template <typename ValueType, typename IndexType>
std::pair<gko::matrix_data<ValueType, IndexType>, gko::dim<2>> generate_stencil(
    std::string stencil_name, const gko::size_type target_local_size)
{
    if (stencil_name == "5pt") {
        return generate_2d_stencil_subdomain<ValueType, IndexType>(
            {1, 1}, {0, 0}, target_local_size, true);
    } else if (stencil_name == "9pt") {
        return generate_2d_stencil_subdomain<ValueType, IndexType>(
            {1, 1}, {0, 0}, target_local_size, false);
    } else if (stencil_name == "7pt") {
        return generate_3d_stencil_subdomain<ValueType, IndexType>(
            {1, 1, 1}, {0, 0, 0}, target_local_size, true);
    } else if (stencil_name == "27pt") {
        return generate_3d_stencil_subdomain<ValueType, IndexType>(
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
 * @see generate_2d_stencil_subdomain
 */
template <typename ValueType, typename IndexType>
std::pair<gko::matrix_data<ValueType, IndexType>, gko::dim<2>>
generate_2d_stencil(gko::experimental::mpi::communicator comm,
                    const gko::size_type target_local_size, bool restricted,
                    bool optimal_comm)
{
    if (optimal_comm) {
        return generate_2d_stencil_subdomain<ValueType, IndexType>(
            {comm.size(), 1}, {comm.rank(), 0}, target_local_size, restricted);
    } else {
        std::array<int, 2> dims{};
        MPI_Dims_create(comm.size(), dims.size(), dims.data());

        std::array<int, 2> coords{};
        coords[0] = comm.rank() % dims[0];
        coords[1] = comm.rank() / dims[0];

        return generate_2d_stencil_subdomain<ValueType, IndexType>(
            dims, coords, target_local_size, restricted);
    }
}


/**
 * Generates matrix data for a given 23 stencil, where the position of this
 * block is given by it's MPI rank.
 *
 * @see generate_3d_stencil_subdomain
 */
template <typename ValueType, typename IndexType>
std::pair<gko::matrix_data<ValueType, IndexType>, gko::dim<2>>
generate_3d_stencil(gko::experimental::mpi::communicator comm,
                    const gko::size_type target_local_size, bool restricted,
                    bool optimal_comm)
{
    if (optimal_comm) {
        return generate_3d_stencil_subdomain<ValueType, IndexType>(
            {comm.size(), 1, 1}, {comm.rank(), 0, 0}, target_local_size,
            restricted);
    } else {
        std::array<int, 3> dims{};

        MPI_Dims_create(comm.size(), dims.size(), dims.data());

        std::array<int, 3> coords{};
        coords[0] = comm.rank() % dims[0];
        coords[1] = (comm.rank() / dims[0]) % dims[1];
        coords[2] = comm.rank() / (dims[0] * dims[1]);

        return generate_3d_stencil_subdomain<ValueType, IndexType>(
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
std::pair<gko::matrix_data<ValueType, IndexType>, gko::dim<2>> generate_stencil(
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
