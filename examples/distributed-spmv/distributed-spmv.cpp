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


// Finally, we need the MPI header for MPI_Init and _Finalize
#include <mpi.h>


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using GlobalIndexType = gko::distributed::global_index_type;
    using LocalIndexType = GlobalIndexType;
    using dist_mtx = gko::distributed::Matrix<ValueType, LocalIndexType>;
    using dist_vec = gko::distributed::Vector<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using part_type = gko::distributed::Partition<LocalIndexType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for
    // an gko::OmpExecutor, which uses OpenMP multi-threading in most of its
    // kernels, a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor and a gko::CudaExecutor which runs the code on a
    // NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    const auto comm = gko::distributed::communicator{};

    std::ifstream a_stream{"data/A.mtx"};
    std::ifstream x_stream{"data/x.mtx"};
    std::ifstream b_stream{"data/b.mtx"};
    auto A_data = gko::read_raw<ValueType, GlobalIndexType>(a_stream);
    auto x_data = gko::read_raw<ValueType, GlobalIndexType>(x_stream);
    auto b_data = gko::read_raw<ValueType, GlobalIndexType>(b_stream);

    // build partition: uniform number of rows per rank
    gko::Array<gko::int64> ranges_array{
        exec->get_master(), static_cast<gko::size_type>(comm.size() + 1)};
    const auto rows_per_rank = A_data.size[0] / comm.size();
    for (int i = 0; i < comm.size(); i++) {
        ranges_array.get_data()[i] = i * rows_per_rank;
    }
    ranges_array.get_data()[comm.size()] = A_data.size[0];
    auto partition =
        gko::share(part_type::build_from_contiguous(exec, ranges_array));

    auto A = dist_mtx::create(exec, comm);
    auto x = dist_vec::create(exec, comm);
    auto b = dist_vec::create(exec, comm);
    A->read_distributed(A_data, partition);
    x->read_distributed(x_data, partition);
    b->read_distributed(b_data, partition);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto minus_one = gko::initialize<vec>({-1.0}, exec);
    A->apply(lend(one), lend(x), lend(minus_one), lend(b));
    auto result = gko::initialize<vec>({0.0}, exec->get_master());
    b->compute_norm2(lend(result));
    std::cout << *result->get_values() << std::endl;

    MPI_Finalize();
}
