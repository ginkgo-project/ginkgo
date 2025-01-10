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
    const gko::size_type target_local_size);


#if GINKGO_BUILD_MPI


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
    const gko::size_type target_local_size, bool optimal_comm);


#endif
#endif  // GINKGO_BENCHMARK_UTILS_STENCIL_MATRIX_HPP
