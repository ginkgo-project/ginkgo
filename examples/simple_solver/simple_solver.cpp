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

/* A bit complicated for now, but this is how to build it:
 *
 * Ginkgo should be compiled with `-DBUILD_REFERENCE=on` option.
 * Once you built ginkgo, go to the build directory and copy the shared
 * libraries located in the following subdirectories:
 *     + core/
 *     + core/device_hooks/
 *     + reference/
 * to this directory.
 *
 * Then compile this file with the following command line:
 *
 * c++ -o test test.cpp -I.. -L. -lginkgo -lginkgo_reference -lginkgo_cpu \
 *     -lginkgo_gpu
 *
 * (if ginkgo was built in debug mode, append 'd' to every library name)
 * Now you should be able to run the program with
 *
 * LD_LIBRARY_PATH=. ./test
 */
#include <include/ginkgo.hpp>
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    // Some shortcuts
    using vec = gko::matrix::Dense<>;
    using mtx = gko::matrix::Dense<>;
    using cg = gko::solver::CgFactory<>;

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "cpu") {
        exec = gko::CpuExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "gpu" &&
               gko::GpuExecutor::get_num_devices() > 0) {
        exec = gko::GpuExecutor::create(0, gko::CpuExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << "[executor]" << std::endl;
        std::exit(-1);
    }

    // Read data
    std::shared_ptr<mtx> A = mtx::create(exec);
    A->read_from_mtx("A.mtx");
    auto b = vec::create(exec);
    b->read_from_mtx("b.mtx");
    auto x = vec::create(exec);
    x->read_from_mtx("x0.mtx");

    // Generate solver
    auto solver_gen = cg::create(exec, 20, 1e-20);
    auto solver = solver_gen->generate(A);

    // Solve system
    solver->apply(b.get(), x.get());

    // Print result
    auto h_x = vec::create(exec->get_master());
    h_x->copy_from(x.get());
    std::cout << "x = [" << std::endl;
    for (int i = 0; i < h_x->get_num_rows(); ++i) {
        std::cout << "    " << h_x->at(i, 0) << std::endl;
    }
    std::cout << "];" << std::endl;

    // Calculate residual
    auto one = vec::create(exec, {1.0});
    auto neg_one = vec::create(exec, {-1.0});
    auto res = vec::create(exec, {0.0});
    A->apply(one.get(), x.get(), neg_one.get(), b.get());
    b->compute_dot(b.get(), res.get());

    auto h_res = vec::create(exec->get_master());
    h_res->copy_from(std::move(res));
    std::cout << "res = " << std::sqrt(h_res->at(0, 0)) << ";" << std::endl;
}
