/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

/*****************************<COMPILATION>***********************************
The easiest way to build the example solver is to use the script provided:
./build.sh <PATH_TO_GINKGO_BUILD_DIR>

Ginkgo should be compiled with `-DBUILD_REFERENCE=on` option.

Alternatively, you can setup the configuration manually:

Go to the <PATH_TO_GINKGO_BUILD_DIR> directory and copy the shared
libraries located in the following subdirectories:

    + core/
    + core/device_hooks/
    + reference/
    + omp/
    + cuda/

to this directory.

Then compile the file with the following command line:

c++ -std=c++11 -o simple_solver simple_solver.cpp -I../.. \
    -L. -lginkgo -lginkgo_reference -lginkgo_omp -lginkgo_cuda

(if ginkgo was built in debug mode, append 'd' to every library name)

Now you should be able to run the program using:

env LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} ./simple_solver

*****************************<COMPILATION>**********************************/

#include <core/test/utils.hpp>
#include <include/ginkgo.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

// Some shortcuts
using vec = gko::matrix::Dense<>;
using mtx = gko::matrix::Coo<>;
using duration_type = std::chrono::microseconds;
std::unique_ptr<vec> gen_mtx(std::shared_ptr<gko::Executor> exec,
                             std::ranlux48 rand_engine, int num_rows,
                             int num_cols, int min_nnz_row)
{
    return gko::test::generate_random_matrix<vec>(
        num_rows, num_cols,
        std::uniform_int_distribution<>(min_nnz_row, num_cols),
        std::normal_distribution<>(-1.0, 1.0), rand_engine, exec);
}

int main(int argc, char *argv[])
{
    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    std::string src_folder;
    std::string mtx_list;
    std::string out_file;
    if (argc >= 4) {
        src_folder = argv[1];
        mtx_list = argv[2];
        out_file = argv[3];
    } else {
        std::cerr << "Usage: " << argv[0]
                  << "src_folder mtx_list out_file [executor]" << std::endl;
        std::exit(-1);
    }
    if (argc == 4 || std::string(argv[4]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 5 && std::string(argv[4]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 5 && std::string(argv[4]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(1, gko::ReferenceExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0]
                  << "src_folder mtx_list out_file [executor]" << std::endl;
        std::exit(-1);
    }

    // Set the testing setting
    constexpr int warm_iter = 2;
    constexpr int test_iter = 10;
    std::ranlux48 rand_engine(42);
    // Open files
    std::ifstream mtx_fd(mtx_list, std::ifstream::in);
    std::ofstream out_fd(out_file, std::ofstream::out);
    out_fd << "name, num_rows, num_cols, nnz, "
              "coo_num_stored_elements, time(us)"
           << std::endl;
    while (!mtx_fd.eof()) {
        duration_type duration(0);
        std::string mtx_file;
        mtx_fd >> mtx_file;
        if (mtx_file.empty()) {
            continue;
        }
        std::cout << src_folder + '/' + mtx_file << std::endl;
        auto data = gko::read_raw<>(src_folder + '/' + mtx_file);
        auto A = mtx::create(exec);
        A->read(data);
        auto x =
            gen_mtx(exec->get_master(), rand_engine, data.size.num_rows, 1, 1);
        auto y =
            gen_mtx(exec->get_master(), rand_engine, data.size.num_cols, 1, 1);
        auto dx = vec::create(exec);
        auto dy = vec::create(exec);
        dx->copy_from(x.get());
        for (int i = 0; i < warm_iter; i++) {
            dy->copy_from(y.get());
            A->apply(dx.get(), dy.get());
        }
        exec->synchronize();
        for (int i = 0; i < test_iter; i++) {
            dy->copy_from(y.get());
            // make sure the executor is finished
            exec->synchronize();
            auto start = std::chrono::system_clock::now();
            A->apply(dx.get(), dy.get());
            // make sure the executor is finished
            exec->synchronize();
            auto finish = std::chrono::system_clock::now();
            duration +=
                std::chrono::duration_cast<duration_type>(finish - start);
        }
        out_fd << mtx_file << ", " << data.size.num_rows << ", "
               << data.size.num_cols << ", " << data.nonzeros.size() << ", "
               << A->get_num_stored_elements() << ", "
               << static_cast<double>(duration.count()) / test_iter
               << std::endl;
    }

    // Close files
    mtx_fd.close();
    out_fd.close();
}
