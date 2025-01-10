// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP
#define GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP


#include <ginkgo/core/base/device_matrix_data.hpp>
#if GINKGO_BUILD_MPI
#include <ginkgo/core/base/mpi.hpp>
#endif


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
gko::device_matrix_data<ValueType, IndexType> generate_2d_stencil_box(
    std::shared_ptr<const gko::Executor> exec, std::array<int, 2> dims,
    std::array<int, 2> positions, const gko::size_type target_local_size,
    bool restricted);


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
gko::device_matrix_data<ValueType, IndexType> generate_3d_stencil_box(
    std::shared_ptr<const gko::Executor> exec, std::array<int, 3> dims,
    std::array<int, 3> positions, const gko::size_type target_local_size,
    bool restricted);

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
gko::device_matrix_data<ValueType, IndexType> generate_stencil(
    std::shared_ptr<const gko::Executor> exec, std::string stencil_name,
    const gko::size_type target_local_size)
{
    if (stencil_name == "5pt") {
        return generate_2d_stencil_box<ValueType, IndexType>(
            exec, {1, 1}, {0, 0}, target_local_size, true);
    } else if (stencil_name == "9pt") {
        return generate_2d_stencil_box<ValueType, IndexType>(
            exec, {1, 1}, {0, 0}, target_local_size, false);
    } else if (stencil_name == "7pt") {
        return generate_3d_stencil_box<ValueType, IndexType>(
            exec, {1, 1, 1}, {0, 0, 0}, target_local_size, true);
    } else if (stencil_name == "27pt") {
        return generate_3d_stencil_box<ValueType, IndexType>(
            exec, {1, 1, 1}, {0, 0, 0}, target_local_size, false);
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
gko::device_matrix_data<ValueType, IndexType> generate_2d_stencil(
    std::shared_ptr<const gko::Executor> exec,
    gko::experimental::mpi::communicator comm,
    const gko::size_type target_local_size, bool restricted, bool optimal_comm)
{
    if (optimal_comm) {
        return generate_2d_stencil_box<ValueType, IndexType>(
            exec, {comm.size(), 1}, {comm.rank(), 0}, target_local_size,
            restricted);
    } else {
        std::array<int, 2> dims{};
        MPI_Dims_create(comm.size(), dims.size(), dims.data());

        std::array<int, 2> coords{};
        coords[0] = comm.rank() % dims[0];
        coords[1] = comm.rank() / dims[0];

        return generate_2d_stencil_box<ValueType, IndexType>(
            exec, dims, coords, target_local_size, restricted);
    }
}


/**
 * Generates matrix data for a given 23 stencil, where the position of this
 * block is given by it's MPI rank.
 *
 * @see generate_3d_stencil_box
 */
template <typename ValueType, typename IndexType>
gko::device_matrix_data<ValueType, IndexType> generate_3d_stencil(
    std::shared_ptr<const gko::Executor> exec,
    gko::experimental::mpi::communicator comm,
    const gko::size_type target_local_size, bool restricted, bool optimal_comm)
{
    if (optimal_comm) {
        return generate_3d_stencil_box<ValueType, IndexType>(
            exec, {comm.size(), 1, 1}, {comm.rank(), 0, 0}, target_local_size,
            restricted);
    } else {
        std::array<int, 3> dims{};

        MPI_Dims_create(comm.size(), dims.size(), dims.data());

        std::array<int, 3> coords{};
        coords[0] = comm.rank() % dims[0];
        coords[1] = (comm.rank() / dims[0]) % dims[1];
        coords[2] = comm.rank() / (dims[0] * dims[1]);

        return generate_3d_stencil_box<ValueType, IndexType>(
            exec, dims, coords, target_local_size, restricted);
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
gko::device_matrix_data<ValueType, IndexType> generate_stencil(
    std::shared_ptr<const gko::Executor> exec, std::string stencil_name,
    gko::experimental::mpi::communicator comm,
    const gko::size_type target_local_size, bool optimal_comm)
{
    if (stencil_name == "5pt") {
        return generate_2d_stencil<ValueType, IndexType>(
            exec, std::move(comm), target_local_size, true, optimal_comm);
    } else if (stencil_name == "9pt") {
        return generate_2d_stencil<ValueType, IndexType>(
            exec, std::move(comm), target_local_size, false, optimal_comm);
    } else if (stencil_name == "7pt") {
        return generate_3d_stencil<ValueType, IndexType>(
            exec, std::move(comm), target_local_size, true, optimal_comm);
    } else if (stencil_name == "27pt") {
        return generate_3d_stencil<ValueType, IndexType>(
            exec, std::move(comm), target_local_size, false, optimal_comm);
    } else {
        throw std::runtime_error("Stencil " + stencil_name +
                                 " not implemented");
    }
}


#endif
#endif  // GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP
