/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

/* TODO:
 * Reset factor of 50
 * Sparse matrix
 * Profiling it properly and look for copy overhead
 * */
#include <fstream>
#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "cuda_profiler_api.h"
template <typename ExecType, typename MatrixType>
void solve_system(ExecType exec, MatrixType system_matrix,
                  gko::matrix::Dense<double> *x,
                  gko::matrix::Dense<double> *rhs, unsigned int krylov_dim,
                  double accuracy)
{
    using gmres = gko::solver::Gmres<double>;
    constexpr unsigned int max_iters = 800;

    // Generate solver
    auto solver_gen =
        gmres::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(max_iters).on(
                    exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(accuracy)
                    .on(exec))
            .with_krylov_dim(krylov_dim)
            .on(exec);
    auto solver = solver_gen->generate(gko::give(system_matrix));

    cudaProfilerStart();

    // Solve system
    solver->apply(rhs, x);

    cudaProfilerStop();
}


bool are_same_mtx(const gko::matrix::Dense<double> *mtx1,
                  const gko::matrix::Dense<double> *mtx2, double error = 1e-12)
{
    auto get_error = [](const double &v1, const double &v2) {
        return std::abs((v1 - v2) / std::max(v1, v2));
    };
    auto size = mtx1->get_size();
    if (size != mtx2->get_size()) {
        std::cerr << "Mismatching sizes!!!\n";
        return false;
    }
    for (int j = 0; j < size[1]; ++j) {
        for (int i = 0; i < size[0]; ++i) {
            if (get_error(mtx1->at(i, j), mtx2->at(i, j)) > error) {
                std::cerr << "Problem at component (" << i << "," << j
                          << "): " << mtx1->at(i, j) << " != " << mtx2->at(i, j)
                          << " !!!\n";
                return false;
            }
            // std::cout << "All good for (" << i << "," << j << "): " <<
            // x->at(i,j) << " == " << x_host->at(i,j) << "\n";
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    using Mtx = gko::matrix::Coo<double>;
    using Dense = gko::matrix::Dense<double>;
    const auto host_ex = gko::ReferenceExecutor::create();
    // const auto omp = gko::OmpExecutor::create();
    //*
    const auto exec = gko::CudaExecutor::create(0, host_ex);
    /*/
    const auto exec = host_ex;
    //*/
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " MatrixMarket_File [krylov_dim]"
                  << std::endl;
        std::exit(-1);
    }

    const unsigned int krylov_dim = argc > 2 ? std::atoi(argv[2]) : 50;

    std::ifstream matrix_stream(argv[1]);
    if (!matrix_stream) {
        std::cerr << "Unable to open the file " << argv[1] << '\n';
        return 1;
    }
    // auto system_matrix =
    // gko::matrix::Coo<double>::read(gko::read(matrix_stream));
    auto system_matrix = gko::read<Mtx>(matrix_stream, exec);
    if (system_matrix->get_size()[0] <= 1 || !system_matrix) {
        std::cerr << "Unable to read the matrix from the file.\n";
        return 1;
    }

    const auto dimensions = system_matrix->get_size();
    if (dimensions[0] != dimensions[1]) {
        std::cerr << "Mismatching dimensions!\n";
        return 1;
    }
    std::cout << "Matrix sucessfully read with dimensions " << dimensions[0]
              << " x " << dimensions[1] << "; Krylov dim: " << krylov_dim
              << std::endl;
    const auto vectorLen = dimensions[0];
    decltype(vectorLen) width = 1;

    std::vector<double> x_vec(width * vectorLen, 1);
    std::vector<double> rhs_vec(width * vectorLen, 1);
    auto x = Dense::create(
        host_ex, gko::dim<2>(vectorLen, width),
        gko::Array<double>::view(host_ex, width * vectorLen, x_vec.data()),
        width);
    auto rhs = Dense::create(
        host_ex, gko::dim<2>(vectorLen, width),
        gko::Array<double>::view(host_ex, width * vectorLen, rhs_vec.data()),
        width);

    auto system_matrix_host = Mtx::create(host_ex);
    system_matrix_host->copy_from(gko::lend(system_matrix));
    auto x_host = Dense::create(host_ex);
    x_host->copy_from(gko::lend(x));
    auto rhs_host = Dense::create(host_ex);
    rhs_host->copy_from(gko::lend(rhs));

    auto backup_mtx = system_matrix->clone();

    solve_system(exec, gko::give(system_matrix), gko::lend(x), gko::lend(rhs),
                 krylov_dim, 1e-12);


    solve_system(host_ex, gko::give(system_matrix_host), gko::lend(x_host),
                 gko::lend(rhs_host), krylov_dim, 1e-12);


    bool error_occured = !are_same_mtx(x.get(), x_host.get());

    auto b_test = rhs->clone();

    backup_mtx->apply(x_host.get(), b_test.get());
    if (are_same_mtx(rhs.get(), b_test.get()))
        std::cout << "Host Implementation seems to be fine!\n";

    std::cout << (error_occured ? "Error occured," : "Successfully")
              << " finished execution!" << std::endl;

    return 0;
}
