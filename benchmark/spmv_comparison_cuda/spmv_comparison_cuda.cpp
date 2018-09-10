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

If you want to check cuda memory first before setting value in the function
`read`, you can use additional_check.patch:

patch -d /path/to/gingko -p1 < additional_check.patch

The output format only support matlab and csv.
The output filename will be `output_prefix`_n`nrhs`.ext
ext is .m for matlab, and .csv for csv.
*****************************<COMPILATION>**********************************/


#include <include/ginkgo.hpp>


#include <cuda_runtime.h>
#include <cusparse.h>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

// #define CHECK
// Some shortcuts
using vec = gko::matrix::Dense<double>;
using mtx = gko::matrix_data<double, gko::int32>;
using duration_type = std::chrono::microseconds;
using csr = gko::matrix::Csr<double, gko::int32>;


class CuspCsrmp : public gko::EnableCreateMethod<CuspCsrmp> {
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

    CuspCsrmp(std::shared_ptr<gko::Executor> exec)
        : csr_(std::move(csr::create(exec))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&desc_));
        ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
    }

    ~CuspCsrmp() noexcept(false)
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

class CuspCsrmm : public gko::EnableCreateMethod<CuspCsrmm> {
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

        ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsrmm(
            handle_, trans_, csr_->get_size()[0], dense_b->get_size()[1],
            csr_->get_size()[1], csr_->get_num_stored_elements(), &alpha, desc_,
            csr_->get_const_values(), csr_->get_const_row_ptrs(),
            csr_->get_const_col_idxs(), db, dense_b->get_size()[0], &beta, dx,
            dense_x->get_size()[0]));
    }

    gko::dim<2> get_size() const noexcept { return csr_->get_size(); }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

    CuspCsrmm(std::shared_ptr<gko::Executor> exec)
        : csr_(std::move(csr::create(exec))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&desc_));
        ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
    }

    ~CuspCsrmm() noexcept(false)
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


class CuspCsrEx : public gko::EnableCreateMethod<CuspCsrEx> {
    using ValueType = double;
    using IndexType = gko::int32;

public:
    void read(const mtx &data)
    {
        csr_->read(data);
        size_t buffer_size;
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();

        ASSERT_NO_CUSPARSE_ERRORS(cusparseCsrmvEx_bufferSize(
            handle_, algmode_, trans_, csr_->get_size()[0], csr_->get_size()[1],
            csr_->get_num_stored_elements(), &alpha, CUDA_R_64F, desc_,
            csr_->get_const_values(), CUDA_R_64F, csr_->get_const_row_ptrs(),
            csr_->get_const_col_idxs(), nullptr, CUDA_R_64F, &beta, CUDA_R_64F,
            nullptr, CUDA_R_64F, CUDA_R_64F, &buffer_size));
        ASSERT_NO_CUDA_ERRORS(cudaMalloc(&buffer_, buffer_size));
        set_buffer_ = true;
    }

    void apply(const gko::LinOp *b, gko::LinOp *x) const
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        auto db = dense_b->get_const_values();
        auto dx = dense_x->get_values();
        auto alpha = gko::one<ValueType>();
        auto beta = gko::zero<ValueType>();
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCsrmvEx(
            handle_, algmode_, trans_, csr_->get_size()[0], csr_->get_size()[1],
            csr_->get_num_stored_elements(), &alpha, CUDA_R_64F, desc_,
            csr_->get_const_values(), CUDA_R_64F, csr_->get_const_row_ptrs(),
            csr_->get_const_col_idxs(), db, CUDA_R_64F, &beta, CUDA_R_64F, dx,
            CUDA_R_64F, CUDA_R_64F, buffer_));
    }

    gko::dim<2> get_size() const noexcept { return csr_->get_size(); }

    gko::size_type get_num_stored_elements() const noexcept
    {
        return csr_->get_num_stored_elements();
    }

    CuspCsrEx(std::shared_ptr<gko::Executor> exec)
        : csr_(std::move(csr::create(exec))),
          trans_(CUSPARSE_OPERATION_NON_TRANSPOSE),
          set_buffer_(false)
    {
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreate(&handle_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&desc_));
        ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST));
#ifdef ALLOWMP
        cusparseAlgMode_t algmode_ = CUSPARSE_ALG_MERGE_PATH;
#endif
    }

    ~CuspCsrEx() noexcept(false)
    {
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroy(handle_));
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDestroyMatDescr(desc_));
        if (set_buffer_) {
            ASSERT_NO_CUDA_ERRORS(cudaFree(buffer_));
        }
    }

private:
    std::shared_ptr<csr> csr_;
    cusparseHandle_t handle_;
    cusparseMatDescr_t desc_;
    cusparseOperation_t trans_;
    cusparseAlgMode_t algmode_;
    void *buffer_;
    bool set_buffer_;
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
        ASSERT_NO_CUSPARSE_ERRORS(cusparseDcsr2hyb(
            handle_, size[0], size[1], desc_, t_csr->get_const_values(),
            t_csr->get_const_row_ptrs(), t_csr->get_const_col_idxs(), hyb_,
            Threshold, Partition));
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

    ~CuspHybrid() noexcept(false)
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

// set matrix shortcuts
using coo = gko::matrix::Coo<double, gko::int32>;
using csri = gko::matrix::Csri<double, gko::int32>;
using ell = gko::matrix::Ell<double, gko::int32>;
using hybrid = gko::matrix::Hybrid<double, gko::int32>;
using sellp = gko::matrix::Sellp<double, gko::int32>;
using cusp_coo = CuspHybrid<CUSPARSE_HYB_PARTITION_USER, 0>;
using cusp_csr = csr;
using cusp_csrex = CuspCsrEx;
using cusp_csrmp = CuspCsrmp;
using cusp_ell = CuspHybrid<CUSPARSE_HYB_PARTITION_MAX, 0>;
using cusp_hybrid = CuspHybrid<>;
using cusp_csrmm = CuspCsrmm;


template <typename RandomEngine>
std::unique_ptr<vec> create_rhs(std::shared_ptr<const gko::Executor> exec,
                                RandomEngine &engine, gko::dim<2> dimension)
{
    auto rhs = vec::create(exec);
    rhs->read(gko::matrix_data<>(
        dimension, std::uniform_real_distribution<>(-1.0, 1.0), engine));
    return rhs;
}

void output(const gko::size_type num, const double val, bool matlab_format,
            std::ofstream &out)
{
    std::string sep = matlab_format ? " " : ", ";
    out << sep << num << sep << val;
}

template <typename MatrixType, bool saved = false, typename... MatrixArgs>
void testing(std::shared_ptr<gko::Executor> exec, const int warm_iter,
             const int test_iter, const mtx &data, vec *x, vec *y,
             bool matlab_format, std::ofstream &out, vec *verified,
             MatrixArgs &&... args)
{
    auto A = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    try {
        A->read(data);
    } catch (...) {
        // -1: read failed
        output(0, -1, matlab_format, out);
        return;
    }
    auto dx = vec::create(exec);
    auto dy = vec::create(exec);
    try {
        dx->copy_from(x);
        dy->copy_from(y);
    } catch (...) {
        // -2 : copy vector failed
        output(0, -2, matlab_format, out);
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
        output(0, -3, matlab_format, out);
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
           static_cast<double>(duration.count()) / test_iter, matlab_format,
           out);
#ifdef CHECK
    if (saved) {
        verified->copy_from(dy.get());
        exec->synchronize();
        std::cerr << "Check" << std::endl;
    } else {
        auto hy = vec::create(exec->get_master());
        hy->copy_from(dy.get());
        exec->synchronize();
        gko::size_type num(0);
        auto res = new double[hy->get_size()[1]];
        auto norm = new double[hy->get_size()[1]];
        for (int j = 0; j < hy->get_size()[1]; j++) {
            res[j] = 0;
            norm[j] = 0;
        }
        for (int i = 0; i < hy->get_size()[0]; i++) {
            for (int j = 0; j < hy->get_size()[1]; j++) {
                res[j] += (hy->at(i, j) - verified->at(i, j)) *
                          (hy->at(i, j) - verified->at(i, j));
                norm[j] += verified->at(i, j) * verified->at(i, j);
            }
        }
        for (int j = 0; j < hy->get_size()[1]; j++) {
            if (std::sqrt(res[j] / norm[j]) > 1e-14) {
                num++;
            }
        }
        delete[] res;
        delete[] norm;
        std::cerr << "failed num = " << num << std::endl;
    }
#endif
}

std::vector<int> split_string(const std::string &str, const char &sep)
{
    std::size_t found;
    std::size_t head = 0;
    std::vector<int> list;
    do {
        found = str.find(sep, head);
        list.emplace_back(std::atoi(str.substr(head, found).c_str()));
        head = found + 1;
    } while (found != std::string::npos);
    return list;
}
int main(int argc, char *argv[])
{
    // Figure out where to run the code
    std::shared_ptr<gko::CudaExecutor> exec;
    std::string src_folder;
    std::string mtx_list;
    std::vector<int> nrhs_list;
    std::string output_prefix;
    int device_id = 0;
    std::vector<std::string> allow_spmv_list{
        "Coo",       "CuspCsr",    "Ell",      "Hybrid",  "Hybrid20",
        "Hybrid40",  "Hybrid60",   "Hybrid80", "Sellp",   "CuspCsrex",
        "CuspCsrmp", "CuspHybrid", "CuspCoo",  "CuspEll", "CuspCsrmm",
        "Csri",      "Csrm",       "Csrc",     "Csr"};
    std::vector<std::string> allow_spmm_list{
        "Coo",      "Ell",      "Hybrid", "Hybrid20", "Hybrid40",
        "Hybrid60", "Hybrid80", "Sellp",  "CuspCsrmm"};
    std::vector<std::string> &allow_list = allow_spmv_list;
    std::vector<std::string> format_list;
    bool matlab_format;
    if (argc >= 7) {
        device_id = std::atoi(argv[1]);
        if (std::string(argv[2]) == "matlab") {
            matlab_format = true;
        } else if (std::string(argv[2]) == "csv") {
            matlab_format = false;
        } else {
            std::cerr << "Do not support the format " << argv[2] << std::endl;
            exit(-1);
        }
        src_folder = argv[3];
        mtx_list = argv[4];
        nrhs_list = split_string(std::string(argv[5]), ',');
        output_prefix = argv[6];
    } else {
        std::cerr << "Usage: " << argv[0] << " "
                  << "device_id format src_folder mtx_list num_rhs_list "
                     "testing_format1 testing_format2 ..."
                  << std::endl;
        std::cerr << "Example: " << argv[0]
                  << " 0 matlab src list 1,4,8,16 Coo CuspCsrmm" << std::endl;

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
    for (const auto &elem : nrhs_list) {
        if (elem > 1) {
            allow_list = allow_spmm_list;
        } else if (elem < 0) {
            std::cerr << "nrhs should be larger than 0" << std::endl;
            exit(-1);
        }
    }
#ifdef CHECK
    format_list.emplace_back("Answer");
#endif
    for (int i = 7; i < argc; i++) {
        if (find(allow_list.begin(), allow_list.end(), std::string(argv[i])) !=
            allow_list.end()) {
            format_list.emplace_back(std::string(argv[i]));
        } else {
            std::cerr << "Unknown format " << argv[i] << std::endl;
        }
    }
    if (format_list.size() == 0) {
        std::cerr << "No available format" << std::endl;
        return 0;
    }
    // Set the testing setting
    const int warm_iter = 2;
    const int test_iter = 10;
    // Open files
    std::ifstream mtx_fd(mtx_list, std::ifstream::in);
    if (mtx_fd.fail()) {
        std::cerr << "The matrix list " << mtx_list << " does not exist."
                  << std::endl;
        std::exit(-1);
    }
    std::vector<std::ofstream> out_fd(nrhs_list.size());
    for (int i = 0; i < nrhs_list.size(); i++) {
        std::string ext = matlab_format ? ".m" : ".csv";
        std::string out_file =
            output_prefix + "_n" + std::to_string(nrhs_list.at(i)) + ext;
        out_fd.at(i).open(out_file, std::ofstream::out);
        if (out_fd.at(i).fail()) {
            std::cerr << "The output file " << out_file << " is failed."
                      << std::endl;
            std::exit(-1);
        }
        std::cout << "The results of nrhs = " << nrhs_list.at(i)
                  << " are stored in the file " << out_file << std::endl;
        if (matlab_format) {
            out_fd.at(i) << "legend_name = { ";
            int n = format_list.size();
            for (int j = 0; j < n; j++) {
                if (j != 0) {
                    out_fd.at(i) << ", ";
                }
                out_fd.at(i) << "'" << format_list.at(j) << "'";
            }
            out_fd.at(i) << " };" << std::endl;

            out_fd.at(i) << "data = [" << std::endl;
            out_fd.at(i)
                << "% #rows #cols #nonzeros (#stored_elements, Spmv_time[us]):";
            for (const auto &elem : format_list) {
                out_fd.at(i) << " " << elem;
            }
            out_fd.at(i) << std::endl;
        } else {
            out_fd.at(i) << "name,#rows,#cols,#nonzeros";
            for (const auto &elem : format_list) {
                out_fd.at(i) << ",#stored_elements_of_" << elem
                             << ",Spmv_time[us]_of_" << elem;
            }
            out_fd.at(i) << std::endl;
        }
    }

    while (!mtx_fd.eof()) {
        std::string mtx_file;
        mtx_fd >> mtx_file;
        if (mtx_file.empty()) {
            continue;
        }
        std::ifstream mtx_fd(src_folder + '/' + mtx_file);
        auto data = gko::read_raw<>(mtx_fd);

        std::string sep(matlab_format ? " " : ", ");
        for (int i = 0; i < nrhs_list.size(); i++) {
            if (matlab_format) {
                out_fd.at(i) << "% " << mtx_file << std::endl;
            } else {
                out_fd.at(i) << mtx_file << ", ";
            }
            int nrhs = nrhs_list.at(i);
            out_fd.at(i) << data.size[0] << sep << data.size[1] << sep
                         << data.nonzeros.size();
            std::ranlux24 rhs_engine(1234);
            auto answer = vec::create(exec->get_master(),
                                      gko::dim<2>{data.size[0], nrhs});
            try {
                vec::create(exec,
                            gko::dim<2>{data.size[0] + data.size[1], nrhs});
            } catch (...) {
                // check GPU memory
                for (const auto &elem : format_list) {
                    output(0, -4, matlab_format, out_fd.at(i));
                }
                out_fd.at(i) << std::endl;
                continue;
            }
            auto x = create_rhs(exec->get_master(), rhs_engine,
                                gko::dim<2>{data.size[1], nrhs});
            auto y = create_rhs(exec->get_master(), rhs_engine,
                                gko::dim<2>{data.size[0], nrhs});
#ifdef CHECK
            std::cerr << "Matrix : " << mtx_file << std::endl;
#endif
            for (const auto &elem : format_list) {
#ifdef CHECK
                std::cerr << "Format : " << elem << " ";
#endif
                if (elem == "Answer") {
                    testing<cusp_csr, true>(exec, warm_iter, test_iter, data,
                                            lend(x), lend(y), matlab_format,
                                            out_fd.at(i), lend(answer));
                } else if (elem == "Coo") {
                    testing<coo>(exec, warm_iter, test_iter, data, lend(x),
                                 lend(y), matlab_format, out_fd.at(i),
                                 lend(answer));
                } else if (elem == "CuspCsr") {
                    testing<cusp_csr>(exec, warm_iter, test_iter, data, lend(x),
                                      lend(y), matlab_format, out_fd.at(i),
                                      lend(answer));
                } else if (elem == "Ell") {
                    testing<ell>(exec, warm_iter, test_iter, data, lend(x),
                                 lend(y), matlab_format, out_fd.at(i),
                                 lend(answer));
                } else if (elem == "Hybrid") {
                    testing<hybrid>(exec, warm_iter, test_iter, data, lend(x),
                                    lend(y), matlab_format, out_fd.at(i),
                                    lend(answer));
                } else if (elem == "Sellp") {
                    testing<sellp>(exec, warm_iter, test_iter, data, lend(x),
                                   lend(y), matlab_format, out_fd.at(i),
                                   lend(answer));
                } else if (elem == "CuspCsrmp") {
                    testing<cusp_csrmp>(exec, warm_iter, test_iter, data,
                                        lend(x), lend(y), matlab_format,
                                        out_fd.at(i), lend(answer));
                } else if (elem == "CuspHybrid") {
                    testing<cusp_hybrid>(exec, warm_iter, test_iter, data,
                                         lend(x), lend(y), matlab_format,
                                         out_fd.at(i), lend(answer));
                } else if (elem == "CuspCoo") {
                    testing<cusp_coo>(exec, warm_iter, test_iter, data, lend(x),
                                      lend(y), matlab_format, out_fd.at(i),
                                      lend(answer));
                } else if (elem == "CuspEll") {
                    testing<cusp_ell>(exec, warm_iter, test_iter, data, lend(x),
                                      lend(y), matlab_format, out_fd.at(i),
                                      lend(answer));
                } else if (elem == "Hybrid20") {
                    testing<hybrid>(
                        exec, warm_iter, test_iter, data, lend(x), lend(y),
                        matlab_format, out_fd.at(i), lend(answer),
                        std::make_shared<hybrid::imbalance_limit>(0.2));
                } else if (elem == "Hybrid40") {
                    testing<hybrid>(
                        exec, warm_iter, test_iter, data, lend(x), lend(y),
                        matlab_format, out_fd.at(i), lend(answer),
                        std::make_shared<hybrid::imbalance_limit>(0.4));
                } else if (elem == "Hybrid60") {
                    testing<hybrid>(
                        exec, warm_iter, test_iter, data, lend(x), lend(y),
                        matlab_format, out_fd.at(i), lend(answer),
                        std::make_shared<hybrid::imbalance_limit>(0.6));
                } else if (elem == "Hybrid80") {
                    testing<hybrid>(
                        exec, warm_iter, test_iter, data, lend(x), lend(y),
                        matlab_format, out_fd.at(i), lend(answer),
                        std::make_shared<hybrid::imbalance_limit>(0.8));
                } else if (elem == "CuspCsrex") {
                    testing<cusp_csrex>(exec, warm_iter, test_iter, data,
                                        lend(x), lend(y), matlab_format,
                                        out_fd.at(i), lend(answer));
                } else if (elem == "CuspCsrmm") {
                    testing<cusp_csrmm>(exec, warm_iter, test_iter, data,
                                        lend(x), lend(y), matlab_format,
                                        out_fd.at(i), lend(answer));
                } else if (elem == "Csri") {
                    testing<csri>(exec, warm_iter, test_iter, data, lend(x),
                                  lend(y), matlab_format, out_fd.at(i),
                                  lend(answer),
                                  std::make_shared<csri::load_balance>(exec));
                } else if (elem == "Csrm") {
                    testing<csri>(exec, warm_iter, test_iter, data, lend(x),
                                  lend(y), matlab_format, out_fd.at(i),
                                  lend(answer),
                                  std::make_shared<csri::merge_path>());
                } else if (elem == "Csrc") {
                    testing<csri>(exec, warm_iter, test_iter, data, lend(x),
                                  lend(y), matlab_format, out_fd.at(i),
                                  lend(answer),
                                  std::make_shared<csri::classical>());
                } else if (elem == "Csr") {
                    testing<csri>(exec, warm_iter, test_iter, data, lend(x),
                                  lend(y), matlab_format, out_fd.at(i),
                                  lend(answer),
                                  std::make_shared<csri::automatical>(exec));
                }
            }
            out_fd.at(i) << std::endl;
        }
    }
    for (int i = 0; i < nrhs_list.size(); i++) {
        if (matlab_format) {
            out_fd.at(i) << "];" << std::endl;
        }
        out_fd.at(i).close();
    }
    // Close files
    mtx_fd.close();
}
