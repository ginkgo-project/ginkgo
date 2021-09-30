/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <chrono>
#include <string>


// Finally, we need the MPI header for MPI_Init and _Finalize
#include <mpi.h>


/**
 * Generates matrix data for a 2D stencil matrix. If restricted is set to true,
 * creates a 5-pt stencil, if it is false creates a 9-pt stencil. If
 * strong_scaling is set to true, creates the same problemsize independent of
 * the number of ranks, if it false the problem size grows with the number of
 * ranks.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_2d_stencil(
    const IndexType dp, std::shared_ptr<gko::mpi::communicator> comm,
    bool restricted, bool strong_scaling)
{
    const auto mat_size = strong_scaling ? dp * dp : dp * dp * comm->size();
    const auto rows_per_rank = gko::ceildiv(mat_size, comm->size());
    const auto start = rows_per_rank * comm->rank();
    const auto end = gko::min(rows_per_rank * (comm->rank() + 1), mat_size);

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
gko::matrix_data<ValueType, IndexType> generate_3d_stencil(
    const IndexType dp, std::shared_ptr<gko::mpi::communicator> comm,
    bool restricted, bool strong_scaling)
{
    const auto mat_size =
        strong_scaling ? dp * dp * dp : dp * dp * dp * comm->size();
    const auto rows_per_rank = gko::ceildiv(mat_size, comm->size());
    const auto start = rows_per_rank * comm->rank();
    const auto end = gko::min(rows_per_rank * (comm->rank() + 1), mat_size);

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


int main(int argc, char* argv[])
{
    const auto fin = gko::mpi::init_finalize(argc, argv);
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using GlobalIndexType = gko::distributed::global_index_type;
    using LocalIndexType = GlobalIndexType;
    using dist_mtx = gko::distributed::Matrix<ValueType>;
    using dist_vec = gko::distributed::Vector<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using part_type = gko::distributed::Partition<gko::int32>;

    const auto comm = gko::mpi::communicator::create_world();
    const auto rank = comm->rank();

    // Print the ginkgo version information.
    if (rank == 0) {
        std::cout << gko::version_info::get() << std::endl;
    }

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " [executor] [DISCRETIZATION_POINTS] [2D] "
                         "[RESTRICT_STENCIL] [STRONG_SCALING]"
                      << std::endl;
            std::cerr << "Default values:" << std::endl;
            std::cerr << "      - executor:        reference" << std::endl;
            std::cerr << "      - DISCRETIZATION_POINTS: 100" << std::endl;
            std::cerr << "      - 2D:                      1" << std::endl;
            std::cerr << "      - RESTRICT_STENCIL:        0" << std::endl;
            std::cerr << "      - STRONG_SCALING:          1" << std::endl;
        }
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [rank] {
                 return gko::CudaExecutor::create(
                     rank, gko::ReferenceExecutor::create(), true);
             }},
            {"hip",
             [rank] {
                 return gko::HipExecutor::create(
                     rank, gko::ReferenceExecutor::create(), true);
             }},
            {"dpcpp",
             [rank] {
                 return gko::DpcppExecutor::create(
                     rank, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    const auto dp = argc >= 3 ? atoi(argv[2]) : 100;
    const bool two_dim = argc >= 4 ? atoi(argv[3]) > 0 : true;
    const bool restricted = argc >= 5 ? atoi(argv[4]) > 0 : false;
    const bool strong_scaling = argc >= 6 ? atoi(argv[5]) > 0 : true;

    // Generate matrix data on each rank
    if (rank == 0) {
        std::cout << "Generating stencil matrix..." << std::endl;
    }
    auto A_data = two_dim ? generate_2d_stencil<ValueType, GlobalIndexType>(
                                dp, comm, restricted, strong_scaling)
                          : generate_3d_stencil<ValueType, GlobalIndexType>(
                                dp, comm, restricted, strong_scaling);
    const auto mat_size = A_data.size[0];
    const auto rows_per_rank = mat_size / comm->size();

    // build partition: uniform number of rows per rank
    gko::Array<gko::int64> ranges_array{
        exec->get_master(), static_cast<gko::size_type>(comm->size() + 1)};
    for (int i = 0; i < comm->size(); i++) {
        ranges_array.get_data()[i] = i * rows_per_rank;
    }
    ranges_array.get_data()[comm->size()] = mat_size;
    auto partition = gko::share(
        part_type::build_from_contiguous(exec->get_master(), ranges_array));

    // Build global matrix from local matrix data.
    auto h_A = dist_mtx::create(exec->get_master(), comm);
    auto A = dist_mtx::create(exec, comm);
    h_A->read_distributed(A_data, partition);
    A->copy_from(h_A.get());

    // Set up global vectors for the distributed SpMV
    if (rank == 0) {
        std::cout << "Setting up vectors..." << std::endl;
    }
    const auto local_size =
        ranges_array.get_data()[rank + 1] - ranges_array.get_data()[rank];
    auto x = dist_vec::create(exec, comm, partition, gko::dim<2>{mat_size, 1},
                              gko::dim<2>{local_size, 1});
    x->fill(gko::one<ValueType>());
    auto b = dist_vec::create(exec, comm, partition, gko::dim<2>{mat_size, 1},
                              gko::dim<2>{local_size, 1});
    b->fill(gko::one<ValueType>());

    // Do a warmup run
    if (rank == 0) {
        std::cout << "Warming up..." << std::endl;
    }
    A->apply(lend(x), lend(b));

    // Do and time the actual benchmark runs
    if (rank == 0) {
        std::cout << "Running benchmark..." << std::endl;
    }
    auto tic = std::chrono::steady_clock::now();
    for (auto i = 0; i < 100; i++) {
        A->apply(lend(x), lend(b));
        exec->synchronize();
    }
    auto toc = std::chrono::steady_clock::now();

    if (rank == 0) {
        std::chrono::duration<double> duration = toc - tic;
        std::cout << "DURATION: " << duration.count() << "s" << std::endl;
    }
}
