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


#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using lower_trs = gko::solver::LowerTrs<ValueType, IndexType>;
    using upper_trs = gko::solver::UpperTrs<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    // Figure out where to run the code
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

    const auto mat_string = argc >= 3 ? argv[2] : "data/A.mtx";
    // Read data
    auto A = gko::share(gko::read<mtx>(std::ifstream(mat_string), exec));
    auto n = A->get_size()[0];
    auto host_b = vec::create(exec->get_master(), gko::dim<2>{n, 1});
    for (auto i = 0; i < n; i++) {
        host_b->at(i, 0) = 1.;
    }
    auto x = vec::create(exec);
    x->copy_from(host_b.get());
    auto b = vec::create_with_config_of(x.get());
    auto y = vec::create_with_config_of(x.get());
    auto z = vec::create_with_config_of(x.get());
    A->apply(x.get(), b.get());
    // Generate LU factorization with GLU
    auto lu_fact =
        gko::factorization::Glu<ValueType, IndexType>::build().on(exec);
    auto lu = lu_fact->generate(A);

    // Extract factors
    auto l = lu->get_l_factor();
    auto u = lu->get_u_factor();

    std::ofstream l_out{"L.mtx"};
    gko::write(l_out, l.get(), gko::layout_type::coordinate);
    std::ofstream u_out{"U.mtx"};
    gko::write(u_out, u.get(), gko::layout_type::coordinate);
    // Apply diagonal scaling and row permutation according to preprocessing
    // done in GLU
    auto row_scal = lu->get_row_scaling();
    auto rp = lu->get_permutation();
    auto piv = lu->get_pivot();
    if (lu->get_mc64_scale()) {
        std::cout << "SCALE" << std::endl;
        for (auto i = 0; i < n; i++) {
            auto p = piv->get_const_data()[i];
            z->at(i, 0) = b->at(rp->get_const_data()[p], 0) *
                          row_scal->get_const_values()[p];
        }
    } else {
        for (auto i = 0; i < n; n++) {
            auto p = piv->get_const_data()[i];
            z->at(i, 0) = b->at(rp->get_const_data()[p]);
        }
    }

    std::ofstream out1{"x1.mtx"};
    gko::write(out1, z.get());

    // Solve the triangular systems
    auto lower = lower_trs::build().on(exec)->generate(l);
    auto upper = upper_trs::build().on(exec)->generate(u);
    lower->apply(gko::lend(z), gko::lend(y));
    std::ofstream out2{"x2.mtx"};
    gko::write(out2, y.get());
    upper->apply(gko::lend(y), gko::lend(z));

    std::ofstream out3{"x3.mtx"};
    gko::write(out3, z.get());
    // Apply final scaling and permutation on solution vector
    auto col_scal = lu->get_col_scaling();
    auto cp = lu->get_inv_permutation();
    if (lu->get_mc64_scale()) {
        for (auto i = 0; i < n; i++) {
            x->at(i, 0) = z->at(cp->get_const_data()[i]) *
                          col_scal->get_const_values()[i];
        }
    } else {
        for (auto i = 0; i < n; i++) {
            x->at(i, 0) = z->at(cp->get_const_data()[i]);
        }
    }

    // Print solution
    std::ofstream x_out{"x.mtx"};
    write(x_out, gko::lend(x));

    // Calculate residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    // A->apply(gko::lend(one), gko::lend(x), gko::lend(neg_one), gko::lend(b));
    // b->compute_norm2(gko::lend(res));
    x->add_scaled(neg_one.get(), host_b.get());
    x->compute_norm2(gko::lend(res));
    std::cout << "Error norm sqrt(r^T r):\n";
    write(std::cout, gko::lend(res));
}
