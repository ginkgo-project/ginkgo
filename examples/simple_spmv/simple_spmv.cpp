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

#include "cuda_runtime.h"
#include "cusparse.h"

// Some shortcuts
using vec = gko::matrix::Dense<double>;
using mtx = gko::matrix_data<double, gko::int32>;

using coo = gko::matrix::Coo<double, gko::int32>;
using ell = gko::matrix::Ell<double, gko::int32>;
using hybrid = gko::matrix::Hybrid<double, gko::int32>;
using csr = gko::matrix::Csr<double, gko::int32>;
using sellp = gko::matrix::Sellp<double, gko::int32>;

using duration_type = std::chrono::microseconds;


class Csrmp : public gko::EnableCreateMethod<Csrmp> {
    using ValueType = double;
    using IndexType = gko::int32;

public:
    void read(const mtx &data) { csr_->read(data); }
    void apply(const gko::LinOp *b, gko::LinOp *x) const
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrmv_mp(
            handle_, trans_, csr_->get_size()[0], csr_->get_size()[1],
            csr_->get_num_stored_elements(), &alpha, desc_,
            csr_->get_const_values(), csr_->get_const_row_ptrs(),
            csr_->get_const_col_idxs(), db, &beta, dx));
    }
    gko::dim<2> get_size() const noexcept { return csr_->get_size(); }
    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

    Csrmp(std::shared_ptr<gko::Executor> exec)
        : csr_(std::move(csr::create(exec))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&desc_));
        ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
    }
    ~Csrmp()
    {
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroy(handle_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyMatDescr(desc_));
    }

private:
    std::shared_ptr<csr> csr_;
    cusparseHandle_t handle_;
    cusparseMatDescr_t desc_;
    cusparseOperation_t trans_;
};
template <cusparseHybPartition_t Partition = CUSPARSE_HYB_PARTITION_AUTO,
          int Threshold = 0>
class CuspHybrid
    : public gko::EnableCreateMethod<CuspHybrid<Partition, Threshold>> {
    using ValueType = double;
    using IndexType = gko::int32;

public:
    gko::dim<2> size;
    void read(const mtx &data)
    {
        auto t_csr = csr::create(exec_);
        t_csr->read(data);
        size[0] = t_csr->get_size()[0];
        size[1] = t_csr->get_size()[1];
        cusparseDcsr2hyb(handle_, size[0], size[1], desc_,
                         t_csr->get_const_values(), t_csr->get_const_row_ptrs(),
                         t_csr->get_const_col_idxs(), hyb_, Threshold,
                         Partition);
    }
    void apply(const gko::LinOp *b, gko::LinOp *x) const
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDhybmv(handle_, trans_, &alpha, desc_,
                                                 hyb_, db, &beta, dx));
    }
    gko::dim<2> get_size() const noexcept { return size; }
    gko::size_type get_num_stored_elements() const noexcept { return 0; }

    CuspHybrid(std::shared_ptr<gko::Executor> exec)
        : exec_(std::move(exec)), trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&desc_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateHybMat(&hyb_));
        ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
    }
    ~CuspHybrid()
    {
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroy(handle_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyMatDescr(desc_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyHybMat(hyb_));
    }

private:
    std::shared_ptr<gko::Executor> exec_;
    cusparseHandle_t handle_;
    cusparseMatDescr_t desc_;
    cusparseOperation_t trans_;
    cusparseHybMat_t hyb_;
};
using cusp_hybrid = CuspHybrid<>;
using cusp_coo = CuspHybrid<CUSPARSE_HYB_PARTITION_USER, 0>;
using cusp_ell = CuspHybrid<CUSPARSE_HYB_PARTITION_MAX, 0>;

template <typename VecType>
void generate_rhs(VecType *rhs)
{
    auto values = rhs->get_values();
    for (gko::size_type row = 0; row < rhs->get_size()[0]; row++) {
        for (gko::size_type col = 0; col < rhs->get_size()[1]; col++) {
            rhs->at(row, col) = static_cast<double>(row) / col;
        }
    }
}

void output(const gko::size_type num, const double val, bool matlab_format)
{
    std::string sep = matlab_format ? " " : ", ";
    std::cout << sep << num << sep << val;
}

template <typename MatrixType>
void testing(std::shared_ptr<gko::Executor> exec, const int warm_iter,
             const int test_iter, const mtx &data, vec *x, vec *y,
             bool matlab_format)
{
    auto A = MatrixType::create(exec);
    try {
        A->read(data);
    } catch (...) {
        // -1: read failed
        output(0, -1, matlab_format);
        return;
    }

    auto dx = vec::create(exec);
    auto dy = vec::create(exec);
    try {
        dx->copy_from(x);
        dy->copy_from(y);
    } catch (...) {
        // -2 : copy vector failed
        output(0, -2, matlab_format);
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
        output(0, -3, matlab_format);
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
    output(A->get_num_stored_elements(),
           static_cast<double>(duration.count()) / test_iter, matlab_format);
}

int main(int argc, char *argv[])
{
    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    std::string src_folder;
    std::string mtx_list;
    int device_id = 0;
    std::string allow_list("Coo;Csr;Ell;Hybrid;Sellp;Csrmp;CuspHybrid;CuspCoo;CuspEll");
    std::vector<std::string> format_list;
    bool matlab_format;
    if (argc >= 5) {
        device_id = std::atoi(argv[1]);
        if (std::string(argv[2]) == "matlab") {
            matlab_format = true;
        } else {
            matlab_format = false;
        }
        src_folder = argv[3];
        mtx_list = argv[4];
    } else {
        std::cerr << "Usage: " << argv[0]
                  << "device_id format src_folder mtx_list testing_format1 "
                     "testing_format2 ..."
                  << std::endl;
        std::exit(-1);
    }
    if (gko::CudaExecutor::get_num_devices() == 0) {
        std::cerr << "This program should be run on gpus" << std::endl;
        exit(-1);
    }

    if (device_id >= 0 && device_id < gko::CudaExecutor::get_num_devices()) {
        exec = gko::CudaExecutor::create(device_id, gko::OmpExecutor::create());
        ASSERT_NO_CUDA_ERRORS(cudaSetDevice(device_id));
    } else {
        std::cerr << "device_id should be in [0, "
                  << gko::CudaExecutor::get_num_devices() << ")." << std::endl;
        std::exit(-1);
    }
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    for (int i = 5; i < argc; i++) {
        if (allow_list.find(argv[i]) != std::string::npos) {
            format_list.emplace_back(std::string(argv[i]));
        } else {
            std::cout << "Unknown format " << argv[i] << std::endl;
        }
    }
    if (format_list.size() == 0) {
        std::cout << "No available format" << std::endl;
        return 0;
    }
    // Set the testing setting
    const int warm_iter = 2;
    const int test_iter = 10;
    // Open files
    std::ifstream mtx_fd(mtx_list, std::ifstream::in);
    if (matlab_format) {
        std::cout << "data = [" << std::endl;
        std::cout
            << "% #rows #cols #nonzeros (#stored_elements, Spmv_time[us]):";
        for (const auto &elem : format_list) {
            std::cout << " " << elem;
        }
        std::cout << std::endl;
    } else {
        std::cout << "name,#rows,#cols,#nonzeros";
        for (const auto &elem : format_list) {
            std::cout << ",#stored_elements_of_" << elem << ",Spmv_time[us]_of_"
                      << elem;
        }
        std::cout << std::endl;
    }
    while (!mtx_fd.eof()) {
        std::string mtx_file;
        mtx_fd >> mtx_file;
        if (mtx_file.empty()) {
            continue;
        }

        auto data = gko::read_raw<>(src_folder + '/' + mtx_file);
        if (matlab_format) {
            std::cout << "% " << mtx_file << std::endl;
        } else {
            std::cout << mtx_file << ", ";
        }
        std::string sep(matlab_format ? " " : ", ");
        std::cout << data.size[0] << sep << data.size[1] << sep
                  << data.nonzeros.size();
        auto x = vec::create(exec->get_master(), gko::dim<2>{data.size[1], 1});
        auto y = vec::create(exec->get_master(), gko::dim<2>{data.size[0], 1});
        generate_rhs(lend(x));
        generate_rhs(lend(y));
        for (const auto &elem : format_list) {
            if (elem == "Coo") {
                testing<coo>(exec, warm_iter, test_iter, data, lend(x), lend(y),
                             matlab_format);
            } else if (elem == "Csr") {
                testing<csr>(exec, warm_iter, test_iter, data, lend(x), lend(y),
                             matlab_format);
            } else if (elem == "Ell") {
                testing<ell>(exec, warm_iter, test_iter, data, lend(x), lend(y),
                             matlab_format);
            } else if (elem == "Hybrid") {
                testing<hybrid>(exec, warm_iter, test_iter, data, lend(x),
                                lend(y), matlab_format);
            } else if (elem == "Sellp") {
                testing<sellp>(exec, warm_iter, test_iter, data, lend(x),
                               lend(y), matlab_format);
            } else if (elem == "Csrmp") {
                testing<Csrmp>(exec, warm_iter, test_iter, data, lend(x),
                               lend(y), matlab_format);
            } else if (elem == "CuspHybrid") {
                testing<cusp_hybrid>(exec, warm_iter, test_iter, data, lend(x),
                               lend(y), matlab_format);
            } else if (elem == "CuspCoo") {
                testing<cusp_coo>(exec, warm_iter, test_iter, data, lend(x),
                               lend(y), matlab_format);
            } else if (elem == "CuspEll") {
                testing<cusp_ell>(exec, warm_iter, test_iter, data, lend(x),
                               lend(y), matlab_format);
            }
        }
        std::cout << std::endl;
    }
    if (matlab_format) {
        std::cout << "];" << std::endl;
    }
    // Close files
    mtx_fd.close();
}
