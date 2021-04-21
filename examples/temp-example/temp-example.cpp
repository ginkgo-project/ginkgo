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


/*
  Temporary file.
  Has to be removed.
*/

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

#include <dirent.h>
#include <memory>


void FillSubdir(const std::string &main_dir, std::vector<std::string> &subdir)
{
    struct dirent *de;  // Pointer for directory entry

    // opendir() returns a pointer of DIR type.
    DIR *dr = opendir(main_dir.c_str());

    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        std::cout << "Could not open " << main_dir << " directory.";
        exit(0);
    }

    // Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html
    // for readdir()
    while ((de = readdir(dr)) != NULL) {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        subdir.push_back(main_dir + std::string(de->d_name));
    }

    closedir(dr);
}


int main(int argc, char *argv[])
{
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

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


    if (argc < 5) {
        printf(
            "\nFormat: ./a.out [executor name] [category name] [Number of "
            "batches] [1 for scaled] \n");
        exit(0);
    }

    const std::string category = argv[2];

    const std::string dir =
        "/home/isha/Desktop/Pele_Matrices_market/" + category + "/";

    std::vector<std::string> subdir{};

    FillSubdir(dir, subdir);

    std::cout << "\nNumber of small problems in one batch of the category - "
              << category << " is: " << subdir.size() << std::endl;

    int problem_size = std::stoi(argv[3]);

    std::cout << "\nSo, the total number of small problems to be solved: "
              << problem_size * subdir.size() << std::endl;

    bool is_scaled = true;

    int scale_info = std::stoi(argv[4]);

    if (scale_info == 1)
        is_scaled = true;
    else
        is_scaled = false;

    std::cout << "\n\nIs Scaling turned on : " << is_scaled << std::endl;

    std::cout << "\n\nStart reading the matrices and their rhs..." << std::endl;

    using ValueType = std::complex<double>;

    using vec = gko::matrix::Dense<ValueType>;
    using csr_mat = gko::matrix::Csr<ValueType, int>;

    std::vector<gko::int32> batch_row_pointers_common;
    std::vector<gko::int32> batch_column_indices_common;
    std::vector<ValueType> batch_values;
    gko::dim<2> size_common;
    int num_nz_common;


    std::vector<std::unique_ptr<vec>> batch_b;
    std::vector<std::unique_ptr<vec>> batch_x;


    const int num_batches = problem_size * subdir.size();

    for (int problem_id = 0; problem_id < num_batches; problem_id++) {
        const std::string subdir_path = subdir[problem_id % subdir.size()];
        std::string file_A;
        std::string file_b;
        if (is_scaled == true) {
            file_A = subdir_path + "/A_scaled.mtx";
            file_b = subdir_path + "/b_scaled.mtx";
        } else {
            file_A = subdir_path + "/A.mtx";
            file_b = subdir_path + "/b.mtx";
        }

        std::unique_ptr<csr_mat> A =
            gko::read<csr_mat>(std::ifstream(file_A), exec->get_master());


        if (problem_id == 0) {
            for (int i = 0; i < A->get_num_stored_elements(); i++) {
                batch_column_indices_common.push_back(
                    A->get_const_col_idxs()[i]);
            }

            for (int i = 0; i < A->get_size()[0] + 1; i++) {
                batch_row_pointers_common.push_back(A->get_const_row_ptrs()[i]);
            }

            size_common = A->get_size();

            num_nz_common = A->get_num_stored_elements();

        } else {
            // GKO_ASSERT_EQ(size_common , A->get_size());

            GKO_ASSERT_EQ(size_common[0], A->get_size()[0]);
            GKO_ASSERT_EQ(size_common[1], A->get_size()[1]);
            GKO_ASSERT_EQ(num_nz_common, A->get_num_stored_elements());

            for (int i = 0; i < size_common[0] + 1; i++) {
                GKO_ASSERT_EQ(batch_row_pointers_common[i],
                              A->get_const_row_ptrs()[i]);
            }

            for (int i = 0; i < num_nz_common; i++) {
                GKO_ASSERT_EQ(batch_column_indices_common[i],
                              A->get_const_col_idxs()[i]);
            }
        }


        for (int i = 0; i < num_nz_common; i++) {
            batch_values.push_back(A->get_const_values()[i]);
        }

        std::unique_ptr<vec> b = gko::read<vec>(std::ifstream(file_b), exec);

        std::unique_ptr<vec> x_cpu =
            vec::create(exec->get_master(), b->get_size(), b->get_stride());

        // set initial guess to 0 for all rhs
        for (int r = 0; r < x_cpu->get_size()[0]; r++) {
            for (int c = 0; c < x_cpu->get_size()[1]; c++) {
                x_cpu->at(r, c) = 0.0;
            }
        }

        std::unique_ptr<vec> x = gko::clone(exec, x_cpu);

        batch_x.push_back(std::move(x));
        batch_b.push_back(std::move(b));
    }


    // Now create batch_csr and batch_dense matrices.
    std::vector<vec *> batch_initial_guess;
    std::vector<vec *> batch_rhs;


    for (int i = 0; i < batch_x.size(); i++) {
        batch_rhs.push_back(batch_b[i].get());
        batch_initial_guess.push_back(batch_x[i].get());
    }

    std::cout << "\n Create batched matrices objects\n";

    // Also convert these vectors: batch_row_pointers , batch_column_indices,
    // batch_values --> to arrays

    gko::Array<gko::int32> row_ptrs_common{exec->get_master(),
                                           batch_row_pointers_common.size()};
    gko::Array<gko::int32> col_idxs_common{exec->get_master(),
                                           batch_column_indices_common.size()};
    gko::Array<ValueType> vals{exec->get_master(), batch_values.size()};

    for (int i = 0; i < row_ptrs_common.get_num_elems(); i++) {
        row_ptrs_common.get_data()[i] = batch_row_pointers_common[i];
    }

    for (int i = 0; i < col_idxs_common.get_num_elems(); i++) {
        col_idxs_common.get_data()[i] = batch_column_indices_common[i];
    }

    for (int i = 0; i < vals.get_num_elems(); i++) {
        vals.get_data()[i] = batch_values[i];
    }


    std::shared_ptr<gko::matrix::BatchCsr<ValueType, gko::int32>> A_csr_batch =
        share(gko::matrix::BatchCsr<ValueType, gko::int32>::create(
            exec, num_batches, size_common, std::move(vals),
            std::move(col_idxs_common), std::move(row_ptrs_common)));


    std::unique_ptr<gko::matrix::BatchDense<ValueType>> x_dense_batch =
        gko::matrix::BatchDense<ValueType>::create(exec, batch_initial_guess);

    std::unique_ptr<gko::matrix::BatchDense<ValueType>> b_dense_batch =
        gko::matrix::BatchDense<ValueType>::create(exec, batch_rhs);

    // x_dense_batch->get_size().check_size_equality(); //not allowed as
    // get_size() retuns a const reference to batch_dim object - part of the
    // batch matrix class (- which is a batchlinop ), and check_size_equality()
    // is a non const member function of batch_dim class
    gko::batch_dim batch_sz_x = x_dense_batch->get_size();
    batch_sz_x.check_size_equality();
    x_dense_batch->set_size(batch_sz_x);

    gko::batch_dim batch_sz_b = b_dense_batch->get_size();
    batch_sz_b.check_size_equality();
    b_dense_batch->set_size(batch_sz_b);


    auto solver_fac =
        gko::solver::BatchBicgstab<ValueType>::build()
            .with_abs_residual_tol(1e-11)
            .with_tolerance_type(gko::stop::batch::ToleranceType::absolute)
            .with_max_iterations(150)
            .with_preconditioner("jacobi")
            .on(exec);


    auto solver = solver_fac->generate(A_csr_batch);

    std::cout << "\n Start solving the batched system \n" << std::endl;

    solver->apply(b_dense_batch.get(), x_dense_batch.get());

    std::cout << std::endl << "Solved" << std::endl;

    std::vector<std::unique_ptr<vec>> vector_of_solution =
        x_dense_batch->unbatch();

    std::cout << "\n\nNow Writing batch of solutions into the files..."
              << std::endl;

    for (int p_id = 0; p_id < subdir.size(); p_id++) {
        const std::string subdir_path = subdir[p_id];

        std::string solution_file;
        if (is_scaled == true) {
            solution_file = subdir_path + "/x_gko_scaled.mtx";
        } else {
            solution_file = subdir_path + "/x_gko.mtx";
        }


        gko::write<vec>(std::ofstream(solution_file),
                        lend(vector_of_solution[p_id]));
    }

    std::cout << "\n Bye!" << std::endl;
}
