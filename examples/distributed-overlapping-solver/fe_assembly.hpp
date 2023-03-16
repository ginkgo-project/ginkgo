#ifndef GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_FE_ASSEMBLY_HPP
#define GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_FE_ASSEMBLY_HPP

#include <ginkgo/ginkgo.hpp>

#include "types.hpp"

template <typename T>
constexpr std::array<std::array<T, 3>, 3> A_loc{
    {{1.0, -0.5, -0.5}, {-0.5, 0.5, 0.0}, {-0.5, 0.0, 0.5}}};


/* divide each quadratic element in to two triangles
 * 0        1
 * |‾‾‾‾‾‾‾/|
 * | utr /  |
 * |  / ltr |
 * |/_______|
 * 2        0
 * The number denote the local index of the vertices.
 * The following two functions create mappings for a specific triangle (x,
 * y) from the local indexing to the global indexing.
 */
template <typename IndexType>
auto create_ltr_map(gko::size_type num_vertices_y,
                    gko::size_type num_vertices_x)
{
    return [&](const auto y, const auto x) {
        std::array<gko::size_type, 3> map{y * num_vertices_x + x + 1,
                                          (y + 1) * num_vertices_x + x + 1,
                                          y * num_vertices_x + x};
        return [=](const auto i) { return static_cast<IndexType>(map[i]); };
    };
}
template <typename IndexType>
auto create_utr_map(gko::size_type num_vertices_y,
                    gko::size_type num_vertices_x)
{
    return [&](const auto y, const auto x) {
        std::array<gko::size_type, 3> map{(y + 1) * num_vertices_x + x,
                                          (y + 1) * num_vertices_x + x + 1,
                                          y * num_vertices_x + x};
        return [=](const auto i) { return static_cast<IndexType>(map[i]); };
    };
}


template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> assemble(gko::size_type num_elements_y,
                                                gko::size_type num_elements_x,
                                                gko::size_type num_vertices_y,
                                                gko::size_type num_vertices_x)
{
    auto utr_map = create_utr_map<IndexType>(num_vertices_y, num_vertices_x);
    auto ltr_map = create_ltr_map<IndexType>(num_vertices_y, num_vertices_x);

    auto process_element = [](auto&& map, auto& data) {
        for (int jy = 0; jy < A_loc<ValueType>.size(); ++jy) {
            for (int jx = 0; jx < A_loc<ValueType>.size(); ++jx) {
                data.add_value(map(jy), map(jx), A_loc<ValueType>[jy][jx]);
            }
        }
    };

    auto process_boundary = [&](const std::vector<int>& local_bdry_idxs,
                                auto&& map, auto& data) {
        for (int i : local_bdry_idxs) {
            auto global_idx = map(i);
            auto global_idx_x = global_idx % num_vertices_x;
            auto global_idx_y = global_idx / num_vertices_x;

            if (global_idx_x != 0) {
                data.set_value(map(i), global_idx - 1, 0.0);
            }
            if (global_idx_x != num_vertices_x - 1) {
                data.set_value(map(i), global_idx + 1, 0.0);
            }
            if (global_idx_y != 0) {
                data.set_value(map(i), global_idx - num_vertices_x, 0.0);
            }
            if (global_idx_y != num_vertices_y - 1) {
                data.set_value(map(i), global_idx + num_vertices_x, 0.0);
            }

            data.set_value(map(i), map(i), 1.0);
        }
    };

    auto size = num_vertices_x * num_vertices_y;
    gko::matrix_assembly_data<ValueType, IndexType> data{
        gko::dim<2>{size, size}};

    for (int iy = 0; iy < num_elements_y; iy++) {
        for (int ix = 0; ix < num_elements_x; ix++) {
            // handle upper triangle
            process_element(utr_map(iy, ix), data);

            // handle lower triangle
            process_element(ltr_map(iy, ix), data);
        }
    }
    for (int iy = 0; iy < num_elements_y; iy++) {
        for (int ix = 0; ix < num_elements_x; ix++) {
            // handle boundary
            if (ix == 0) {
                process_boundary({0, 2}, utr_map(iy, ix), data);
            }
            if (ix == num_elements_x - 1) {
                process_boundary({0, 1}, ltr_map(iy, ix), data);
            }
            if (iy == 0) {
                process_boundary({0, 2}, ltr_map(iy, ix), data);
            }
            if (iy == num_elements_y - 1) {
                process_boundary({0, 1}, utr_map(iy, ix), data);
            }
        }
    }

    return data.get_ordered_data();
}


// u(0) = u(1) = 1
// values in the interior will be overwritten during the communication
// also set initial guess to dirichlet condition
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> assemble_rhs(
    gko::size_type num_vertices_y, gko::size_type num_vertices_x,
    bool left_brdy, bool right_brdy)
{
    auto f_one = [&](const auto iy, const auto ix) { return 1.0; };
    auto f_linear = [&](const auto iy, const auto ix) {
        return 0.5 * (ix / (num_vertices_x - 1) + iy / (num_vertices_y - 1));
    };

    auto size = num_vertices_x * num_vertices_y;
    gko::matrix_assembly_data<ValueType, IndexType> data{gko::dim<2>{size, 1}};

    // vertical boundaries
    for (int i = 0; i < num_vertices_y; i++) {
        if (left_brdy) {
            auto idx = i * num_vertices_x;
            data.set_value(idx, 0, f_one(i, 0));
        }
        if (right_brdy) {
            auto idx = (i + 1) * num_vertices_x - 1;
            data.set_value(idx, 0, f_one(i, num_vertices_x - 1));
        }
    }
    // horizontal boundaries
    for (int i = 0; i < num_vertices_x; i++) {
        {
            auto idx = i;
            data.set_value(idx, 0, f_one(0, i));
        }
        {
            auto idx = i + (num_vertices_y - 1) * num_vertices_x;
            data.set_value(idx, 0, f_one(num_vertices_y - 1, i));
        }
    }
    return data.get_ordered_data();
}


std::vector<shared_idx_t> setup_shared_idxs(gko::size_type num_elements_y,
                                            gko::size_type num_elements_x,
                                            int overlap_size,
                                            bool share_left_bdry,
                                            bool share_right_bdry,
                                            int this_rank)
{
    std::vector<shared_idx_t> shared_idxs(
        share_right_bdry * (num_elements_y + 1) +
        share_left_bdry * (num_elements_y + 1));
    auto utr_map = create_utr_map<int>(num_elements_y + 1, num_elements_x + 1);
    auto ltr_map = create_ltr_map<int>(num_elements_y + 1, num_elements_x + 1);
    // TODO: should remove physical boundary idxs
    auto fixed_x_map = [&](const auto x, auto&& map) {
        return [=](const auto y) { return map(y, x); };
    };
    auto setup_idxs = [num_elements_y](
                          auto&& partial_map_local, auto&& partial_map_remote,
                          int remote_rank,
                          const std::vector<int> element_local_bdry_idx,
                          shared_idx_t* idxs) {
        for (int iy = 0; iy < num_elements_y; ++iy) {
            auto local_map = partial_map_local(iy);
            auto remote_map = partial_map_remote(iy);
            if (iy == 0) {
                idxs[iy] = {local_map(element_local_bdry_idx[0]),
                            remote_map(element_local_bdry_idx[0]), remote_rank,
                            remote_rank};
            }
            idxs[iy + 1] = idxs[iy] = {local_map(element_local_bdry_idx[1]),
                                       remote_map(element_local_bdry_idx[1]),
                                       remote_rank, remote_rank};
        }
    };
    if (share_left_bdry) {
        setup_idxs(fixed_x_map(0, utr_map),
                   fixed_x_map(num_elements_x - 1 - overlap_size, utr_map),
                   this_rank - 1, {0, 1}, shared_idxs.data());
    }
    if (share_right_bdry) {
        auto offset = share_left_bdry * (num_elements_y + 1);

        setup_idxs(fixed_x_map(num_elements_x - 1, ltr_map),
                   fixed_x_map(overlap_size, ltr_map), this_rank + 1, {0, 1},
                   shared_idxs.data() + offset);
    }
    return shared_idxs;
}

#endif  // GINKGO_EXAMPLES_DISTRIBUTED_OVERLAPPING_SOLVER_FE_ASSEMBLY_HPP
