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


    // batch of four 3X3 matrices
    /*
        matrix A

        1 3 0        1 4  0        1 3 0       1 2 0
        5 0 9        5 0 10        1 0 1       5 0 2
        0 0 1        0 0  1        0 0 1       0 0 2


    */

    /*
        matrix b

        0           1               1               1
        1.2         1.2             1.2             1
        -5          -5              1               1
    */

    /* matrix x

        0           0               0               0
        0           0               0               0
        0           0               0               0

    */
    /*
     using gko::size_type;

     gko::Array<gko::int32> row_ptrs{exec->get_master(), 4,
                                     new gko::int32[4]{0, 2, 4, 5}};

     gko::Array<gko::int32> col_idxs{exec->get_master(), 5,
                                     new gko::int32[5]{0, 1, 0, 2, 2}};

     gko::Array<double> vals{exec->get_master(), 20,
                             new double[20]{1, 3, 5, 9, 1, 1, 4, 5, 10, 1,
                                            1, 3, 1, 1, 1, 1, 2, 5, 2,  2}};

     gko::Array<double> b_vals{
         exec->get_master(), 12,
         new double[12]{0, 1.2, -5, 1, 1.2, -5, 1, 1.2, 1, 1, 1, 1}};

     gko::Array<double> x_vals{
         exec->get_master(), 12,
         new double[12]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};


     row_ptrs.set_executor(exec);
     col_idxs.set_executor(exec);
     vals.set_executor(exec);

     std::shared_ptr<gko::matrix::BatchCsr<double, gko::int32>> A =
         gko::share(gko::matrix::BatchCsr<double>::create(
             exec, 4, gko::dim<2>{3, 3}, std::move(vals), std::move(col_idxs),
             std::move(row_ptrs)));

     std::unique_ptr<gko::matrix::BatchDense<double>> b =
         gko::matrix::BatchDense<double>::create(
             exec,
             std::vector<gko::dim<2>>{gko::dim<2>{3, 1}, gko::dim<2>{3, 1},
                                      gko::dim<2>{3, 1}, gko::dim<2>{3, 1}},
             std::move(b_vals), std::vector<gko::size_type>{1, 1, 1, 1});


     std::unique_ptr<gko::matrix::BatchDense<double>> x =
         gko::matrix::BatchDense<double>::create(
             exec,
             std::vector<gko::dim<2>>{gko::dim<2>{3, 1}, gko::dim<2>{3, 1},
                                      gko::dim<2>{3, 1}, gko::dim<2>{3, 1}},
             std::move(x_vals), std::vector<gko::size_type>{1, 1, 1, 1});


     auto fac = gko::solver::BatchBicgstab<double>::build()
                    .with_rel_residual_tol(1e-11)
                    .with_max_iterations(150)
                    .on(exec);


     // auto fac =
     gko::solver::BatchRichardson<double>::build().with_rel_residual_tol(1e-11).with_max_iterations(150).on(exec);

     auto solver = fac->generate(A);

     solver->apply(b.get(), x.get());


     std::cout << std::endl;

     auto x_cpu = gko::clone(exec->get_master(), x);

     const double *cpu_arr = x_cpu->get_const_values();

     for (int i = 0; i < 12; i++) {
         std::cout << i << "  :  " << cpu_arr[i] << std::endl;
     }

     */


    if (argc < 5) {
        printf(
            "\nFormat: ./a.out [executor name] [category name] [problemsize] "
            "[1 for scaled] \n");
        exit(0);
    }

    const std::string category = argv[2];


    const std::string dir =
        "/home/isha/Desktop/Pele_Matrices_market/" + category + "/";

    std::vector<std::string> subdir{};

    FillSubdir(dir, subdir);

    std::cout << "\nNumber of small problems in category - " << category
              << " is: " << subdir.size() << std::endl;

    int Problem_Size = std::stoi(argv[3]);

    std::cout << "\nSo, the number of small problems to be solved: "
              << Problem_Size * subdir.size() << std::endl;

    bool is_scaled = true;

    int scale_info = std::stoi(argv[3]);

    if (scale_info == 1)
        is_scaled = true;
    else
        is_scaled = false;

    std::cout << "\n\nStart reading the matrices and their rhs..." << std::endl;

    std::vector<gko::int32> batch_row_pointers;
    std::vector<gko::int32> batch_column_indices;
    std::vector<double> batch_values;
    gko::dim<2> sz;

    using vec = gko::matrix::Dense<double>;
    using csr_mat = gko::matrix::Csr<double, int>;

    std::vector<std::unique_ptr<vec>> batch_b;
    std::vector<std::unique_ptr<vec>> batch_x;

    const int num_batches = Problem_Size * subdir.size();

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
                batch_column_indices.push_back(A->get_const_col_idxs()[i]);
            }

            for (int i = 0; i < A->get_size()[0] + 1; i++) {
                batch_row_pointers.push_back(A->get_const_row_ptrs()[i]);
            }

            sz = A->get_size();

        } else {
            // GKO_ASSERT_EQ(sz , A->get_size());

            GKO_ASSERT_EQ(sz[0], A->get_size()[0]);
            GKO_ASSERT_EQ(sz[1], A->get_size()[1]);
            GKO_ASSERT_EQ(batch_column_indices.size(),
                          A->get_num_stored_elements());

            for (int i = 0; i < sz[0] + 1; i++) {
                GKO_ASSERT_EQ(batch_row_pointers[i],
                              A->get_const_row_ptrs()[i]);
            }

            for (int i = 0; i < batch_column_indices.size(); i++) {
                GKO_ASSERT_EQ(batch_column_indices[i],
                              A->get_const_col_idxs()[i]);
            }
        }


        for (int i = 0; i < A->get_num_stored_elements(); i++) {
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


    // Also convert these vectors: batch_row_pointers , batch_column_indices,
    // batch_values --> to arrays

    gko::Array<gko::int32> row_ptrs{exec->get_master(),
                                    batch_row_pointers.size()};
    gko::Array<gko::int32> col_idxs{exec->get_master(),
                                    batch_column_indices.size()};
    gko::Array<double> vals{exec->get_master(), batch_values.size()};

    for (int i = 0; i < row_ptrs.get_num_elems(); i++) {
        row_ptrs.get_data()[i] = batch_row_pointers[i];
    }

    for (int i = 0; i < col_idxs.get_num_elems(); i++) {
        col_idxs.get_data()[i] = batch_column_indices[i];
    }

    for (int i = 0; i < vals.get_num_elems(); i++) {
        vals.get_data()[i] = batch_values[i];
    }


    std::shared_ptr<gko::matrix::BatchCsr<double, gko::int32>> A_csr_batch =
        share(gko::matrix::BatchCsr<double, int>::create(
            exec, num_batches, sz, std::move(vals), std::move(col_idxs),
            std::move(row_ptrs)));


    std::unique_ptr<gko::matrix::BatchDense<double>> x_dense_batch =
        gko::matrix::BatchDense<double>::create(exec, batch_initial_guess);


    std::unique_ptr<gko::matrix::BatchDense<double>> b_dense_batch =
        gko::matrix::BatchDense<double>::create(exec, batch_rhs);


    auto solver_fac = gko::solver::BatchBicgstab<double>::build()
                          .with_rel_residual_tol(1e-11)
                          .with_max_iterations(150)
                          .with_preconditioner("jacobi")
                          .on(exec);


    // std::cout << A_csr_batch->get_num_stored_elements() << std::endl;
    // std::cout << "r: " << A_csr_batch->get_batch_sizes()[0][0]
    //           << "    c:" << A_csr_batch->get_batch_sizes()[0][1] <<
    //           std::endl;
    // std::cout << A_csr_batch->get_num_batches() << std::endl;

    auto solver = solver_fac->generate(A_csr_batch);

    solver->apply(b_dense_batch.get(), x_dense_batch.get());

    std::cout << std::endl << "Solved" << std::endl;

    // auto x_cpu = gko::clone(exec->get_master(), x_dense_batch);

    std::vector<std::unique_ptr<vec>> vector_of_solution =
        x_dense_batch->unbatch();

    std::cout << "\n\nNow Write solution into the file" << std::endl;

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
