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


#include <include/ginkgo.hpp>

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>


// Some shortcuts
using vec = gko::matrix::Dense<double>;
using mtx = gko::matrix_data<double, gko::int32>;

using coo = gko::matrix::Coo<double, gko::int32>;
using ell = gko::matrix::Ell<double, gko::int32>;
using hybrid = gko::matrix::Hybrid<double, gko::int32>;
using csr = gko::matrix::Csr<double, gko::int32>;
using sellp = gko::matrix::Sellp<double, gko::int32>;

using duration_type = std::chrono::microseconds;
template <typename VecType>
void generate_rhs(VecType *rhs)
{
    auto values = rhs->get_values();
    for (gko::size_type row = 0; row < rhs->get_size().num_rows; row++) {
        for (gko::size_type col = 0; col < rhs->get_size().num_cols; col++) {
            rhs->at(row, col) = static_cast<double>(row) / col;
        }
    }
}
template <bool line_end = false, bool matlab_format = true>
void output(const gko::size_type num, const double val)
{
    std::string sep = matlab_format ? " " : ", ";
    std::cout << num << sep << val;
    if (line_end) {
        std::cout << std::endl;
    } else {
        std::cout << sep;
    }
}
template <bool line_end = false, bool matlab_format = true, typename MatrixType>
void testing(std::shared_ptr<gko::Executor> exec, const int warm_iter,
             const int test_iter, const mtx &data, vec *x, vec *y)
{
    auto A = MatrixType::create(exec);
    try {
        A->read(data);
    } catch (...) {
        // -1: read failed
        output<line_end, matlab_format>(0, -1);
        return;
    }

    auto dx = vec::create(exec);
    auto dy = vec::create(exec);
    try {
        dx->copy_from(x);
        dy->copy_from(y);
    } catch (...) {
        // -2 : copy vector failed
        output<line_end, matlab_format>(0, -2);
        return;
    }

    // warm up
    try {
        for (int i = 0; i < warm_iter; i++) {
            dy->copy_from(y);
            A->apply(dx.get(), dy.get());
        }
        exec->synchronize();
    } catch (...) {
        // -3 : apply failed
        output<line_end, matlab_format>(0, -3);
        return;
    }
    // Test
    duration_type duration(0);
    for (int i = 0; i < test_iter; i++) {
        dy->copy_from(y);
        // make sure copy is finished
        exec->synchronize();
        auto start = std::chrono::system_clock::now();
        A->apply(dx.get(), dy.get());
        // make sure apply is finished
        exec->synchronize();
        auto finish = std::chrono::system_clock::now();
        duration += std::chrono::duration_cast<duration_type>(finish - start);
    }
    output<line_end, matlab_format>(
        A->get_num_stored_elements(),
        static_cast<double>(duration.count()) / test_iter);
}

int main(int argc, char *argv[])
{
    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    std::string src_folder;
    std::string mtx_list;
    std::string format_list;
    if (argc >= 3) {
        src_folder = argv[1];
        mtx_list = argv[2];
    } else {
        std::cerr << "Usage: " << argv[0] << "src_folder mtx_list [executor]"
                  << std::endl;
        std::exit(-1);
    }
    if (argc == 3 || std::string(argv[3]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 4 && std::string(argv[3]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 4 && std::string(argv[3]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(1, gko::OmpExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << "src_folder mtx_list [executor]"
                  << std::endl;
        std::exit(-1);
    }

    // Set the testing setting
    const int warm_iter = 2;
    const int test_iter = 10;
    // Open files
    std::ifstream mtx_fd(mtx_list, std::ifstream::in);
    std::cout << "name,num_rows,num_cols,nnz,"
                 "total_num_stored_elements,time(us)"
              << std::endl;
    while (!mtx_fd.eof()) {
        duration_type duration(0);
        std::string mtx_file;
        mtx_fd >> mtx_file;
        if (mtx_file.empty()) {
            continue;
        }

        auto data = gko::read_raw<>(src_folder + '/' + mtx_file);
        std::cout << mtx_file << ", " << data.size.num_rows << ", "
                  << data.size.num_cols << ", " << data.nonzeros.size() << ", ";
        auto x =
            vec::create(exec->get_master(), gko::dim{data.size.num_cols, 1});
        auto y =
            vec::create(exec->get_master(), gko::dim{data.size.num_rows, 1});
        generate_rhs(lend(x));
        generate_rhs(lend(y));
        for (int i = 0; i < 5; i++) {
            switch (i) {
            case 0: {
                // auto mat = coo::create(exec);
                testing<false, true, coo>(exec, warm_iter, test_iter, data,
                                          lend(x), lend(y));
                break;
            }
            case 1: {
                // auto mat = csr::create(exec);
                testing<false, true, csr>(exec, warm_iter, test_iter, data,
                                          lend(x), lend(y));
                break;
            }
            case 2: {
                // auto mat = ell::create(exec);
                testing<false, true, ell>(exec, warm_iter, test_iter, data,
                                          lend(x), lend(y));
                break;
            }
            case 3: {
                // auto mat = hybrid::create(exec);
                testing<false, true, hybrid>(exec, warm_iter, test_iter, data,
                                             lend(x), lend(y));
                break;
            }
            case 4: {
                // auto mat = sellp::create(exec);
                testing<true, true, sellp>(exec, warm_iter, test_iter, data,
                                           lend(x), lend(y));
                break;
            }
            }
        }
    }

    // Close files
    mtx_fd.close();
}
