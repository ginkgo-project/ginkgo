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
#include <string>


int main(int argc, char* argv[])
{
    const gko::mpi::environment env(argc, argv);
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using GlobalIndexType = gko::distributed::global_index_type;
    using LocalIndexType = GlobalIndexType;
    using dist_mtx = gko::distributed::Matrix<ValueType, LocalIndexType>;
    using dist_vec = gko::distributed::Vector<ValueType, LocalIndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using part_type = gko::distributed::Partition<LocalIndexType>;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 2);

    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(
                     0, gko::ReferenceExecutor::create(), true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(
                     0, gko::ReferenceExecutor::create(), true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    const auto comm = std::make_shared<gko::mpi::communicator>(MPI_COMM_WORLD);
    const auto num_rows = grid_dim * grid_dim * grid_dim;

    gko::matrix_data<ValueType, GlobalIndexType> A_data;
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    A_data.size = {num_rows, num_rows};
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};
    for (int i = 0; i < grid_dim; i++) {
        for (int j = 0; j < grid_dim; j++) {
            for (int k = 0; k < grid_dim; k++) {
                auto idx = i * grid_dim * grid_dim + j * grid_dim + k;
                if (i > 0)
                    A_data.nonzeros.emplace_back(idx, idx - grid_dim * grid_dim,
                                                 -1);
                if (j > 0)
                    A_data.nonzeros.emplace_back(idx, idx - grid_dim, -1);
                if (k > 0) A_data.nonzeros.emplace_back(idx, idx - 1, -1);
                A_data.nonzeros.emplace_back(idx, idx, 8);
                if (k < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + 1, -1);
                if (j < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + grid_dim, -1);
                if (i < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + grid_dim * grid_dim,
                                                 -1);
                b_data.nonzeros.emplace_back(
                    idx, 0, std::sin(i * 0.01 + j * 0.14 + k * 0.056));
                x_data.nonzeros.emplace_back(idx, 0, 1.0);
            }
        }
    }

    // build partition: uniform number of rows per rank
    gko::Array<gko::int64> ranges_array{
        exec->get_master(), static_cast<gko::size_type>(comm->size() + 1)};
    const auto rows_per_rank = num_rows / comm->size();
    for (int i = 0; i < comm->size(); i++) {
        ranges_array.get_data()[i] = i * rows_per_rank;
    }
    ranges_array.get_data()[comm->size()] = num_rows;
    auto partition = gko::share(
        part_type::build_from_contiguous(exec->get_master(), ranges_array));

    auto A_host =
        gko::share(dist_mtx::create(exec->get_master(), A_data.size, comm));
    A_host->read_distributed(A_data, partition);
    auto A = gko::share(dist_mtx::create(exec, A_data.size, comm));
    A->copy_from(A_host.get());
    auto B = gko::share(dist_mtx::create(exec, A_data.size, comm));
    B->copy_from(A_host.get());
    auto X = dist_mtx::create(exec, gko::dim<2>(A_data.size[0], A_data.size[1]),
                              partition, comm);
    auto x = dist_vec::create(
        exec, comm, partition, gko::dim<2>(A_data.size[0], 1),
        gko::dim<2>(partition->get_part_size(comm->rank()), 1));
    auto x_spgemm = dist_vec::create(
        exec, comm, partition, gko::dim<2>(A_data.size[0], 1),
        gko::dim<2>(partition->get_part_size(comm->rank()), 1));
    auto b = dist_vec::create(
        exec, comm, partition, gko::dim<2>(A_data.size[0], 1),
        gko::dim<2>(partition->get_part_size(comm->rank()), 1));
    auto y = dist_vec::create(
        exec, comm, partition, gko::dim<2>(A_data.size[0], 1),
        gko::dim<2>(partition->get_part_size(comm->rank()), 1));
    y->fill(gko::zero<ValueType>());
    b->fill(gko::one<ValueType>());
    x->fill(gko::zero<ValueType>());
    x_spgemm->fill(gko::zero<ValueType>());
    A->read_distributed(A_data, partition);
    B->read_distributed(A_data, partition);
    std::cout << " Rank " << comm->rank() << " Mat size " << A->get_size()
              << " lmat size " << A->get_local_diag()->get_size() << std::endl;

    A->apply(lend(B), lend(X));
    std::cout << " Rank " << comm->rank() << " X Mat size " << X->get_size()
              << " lmat size " << X->get_local_diag()->get_size()
              << " offdiag mat size " << X->get_local_offdiag()->get_size()
              << std::endl;
    std::cout << " Rank " << comm->rank() << " b Vec size " << b->get_size()
              << std::endl;
    std::cout << " Rank " << comm->rank() << " x spgemm Vec size "
              << x_spgemm->get_size() << std::endl;

    B->apply(lend(b), lend(y));

    std::cout << "Here " << __LINE__ << std::endl;
    A->apply(lend(y), lend(x));
    std::cout << "Here " << __LINE__ << std::endl;
    X->apply2(lend(b), lend(x_spgemm));
    std::cout << "Here " << __LINE__ << std::endl;
    auto one = gko::initialize<vec>({1.0}, exec);
    auto minus_one = gko::initialize<vec>({-1.0}, exec);
    x_spgemm->add_scaled(lend(minus_one), lend(x));
    auto result = gko::initialize<vec>({0.0}, exec->get_master());
    x_spgemm->compute_norm2(lend(result));
    std::cout << "Spgemm error: " << *result->get_values() << std::endl;
}
