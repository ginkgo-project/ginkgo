// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "benchmark/utils/stencil_matrix_kernels.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace stencil {

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


template <typename ValueType, typename IndexType>
gko::device_matrix_data<ValueType, IndexType> generate_2d_stencil_box(
    std::shared_ptr<const gko::Executor> exec, std::array<int, 2> dims,
    std::array<int, 2> positions, const gko::size_type target_local_size,
    bool restricted)
{
    auto num_boxes =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});

    const auto discretization_points =
        static_cast<IndexType>(closest_nth_root(target_local_size, 2));
    const auto local_size = static_cast<gko::size_type>(discretization_points *
                                                        discretization_points);
    const auto global_size = local_size * num_boxes;

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

    auto row_idxs = A_data.get_row_idxs();
    auto col_idxs = A_data.get_col_idxs();
    auto vals = A_data.get_values();

    for (IndexType iy = 0; iy < discretization_points; ++iy) {
        for (IndexType ix = 0; ix < discretization_points; ++ix) {
            auto row = flat_idx(iy, ix);
            for (IndexType dy : {-1, 0, 1}) {
                for (IndexType dx : {-1, 0, 1}) {
                    if (is_valid_neighbor(dy, dx)) {
                        auto col = flat_idx(iy + dy, ix + dx);
                        if (is_in_box(col,
                                      static_cast<IndexType>(global_size))) {
                            auto nnz_idx = row * 9 + (dx + 1) + 3 * (dy + 1);
                            row_idxs[nnz_idx] = row;
                            col_idxs[nnz_idx] = col;
                            if (col != row) {
                                vals[nnz_idx] = -gko::one<ValueType>();
                            } else {
                                vals[nnz_idx] = diag_value;
                            }
                        }
                    }
                }
            }
        }
    }

    A_data.remove_zeros();
    return {exec, A_data};
}


template <typename ValueType, typename IndexType>
gko::device_matrix_data<ValueType, IndexType> generate_3d_stencil_box(
    std::shared_ptr<const gko::Executor> exec, std::array<int, 3> dims,
    std::array<int, 3> positions, const gko::size_type target_local_size,
    bool restricted)
{
    auto num_boxes =
        std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});

    const auto discretization_points =
        static_cast<IndexType>(closest_nth_root(target_local_size, 3));
    const auto local_size = static_cast<gko::size_type>(
        discretization_points * discretization_points * discretization_points);
    const auto global_size = local_size * num_boxes;

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

    auto A_data = gko::device_matrix_data<ValueType, IndexType>(
        exec->get_master(),
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)},
        local_size * 27);
    A_data.fill_zero();

    auto row_idxs = A_data.get_row_idxs();
    auto col_idxs = A_data.get_col_idxs();
    auto vals = A_data.get_values();

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
                                    auto nnz_idx = row * 27 + (dx + 1) +
                                                   3 * (dy + 1) + 9 * (dz + 1);
                                    row_idxs[nnz_idx] = row;
                                    col_idxs[nnz_idx] = col;
                                    if (col != row) {
                                        vals[nnz_idx] = -gko::one<ValueType>();
                                    } else {
                                        vals[nnz_idx] = diag_value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    A_data.remove_zeros();
    return {exec, A_data};
}

}  // namespace stencil
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
