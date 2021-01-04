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

#ifndef GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_
#define GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_


#include <ginkgo/ginkgo.hpp>


#include <map>
#include <string>


#include <gflags/gflags.h>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/overhead_linop.hpp"


DEFINE_string(
    preconditioners, "none",
    "A comma-separated list of preconditioners to use. "
    "Supported values are: none, jacobi, parict, parilu, parilut, ilu, "
    "parict-isai, parilu-isai, parilut-isai, ilu-isai, overhead");

DEFINE_uint32(parilu_iterations, 5,
              "The number of iterations for ParICT/ParILU(T)");

DEFINE_bool(parilut_approx_select, true,
            "Use approximate selection for ParICT/ParILUT");

DEFINE_double(parilut_limit, 2.0, "The fill-in limit for ParICT/ParILUT");

DEFINE_int32(
    isai_power, 1,
    "Which power of the sparsity structure to use for ISAI preconditioners");

DEFINE_string(jacobi_storage, "0,0",
              "Defines the kind of storage optimization to perform on "
              "preconditioners that support it. Supported values are: "
              "autodetect and <X>,<Y> where <X> and <Y> are the input "
              "parameters used to construct a precision_reduction object.");

DEFINE_double(jacobi_accuracy, 1e-1,
              "This value is used as the accuracy flag of the adaptive Jacobi "
              "preconditioner.");

DEFINE_uint32(jacobi_max_block_size, 32,
              "Maximal block size of the block-Jacobi preconditioner");


// parses the Jacobi storage optimization command line argument
gko::precision_reduction parse_storage_optimization(const std::string &flag)
{
    if (flag == "autodetect") {
        return gko::precision_reduction::autodetect();
    }
    const auto parts = split(flag, ',');
    if (parts.size() != 2) {
        throw std::runtime_error(
            "storage_optimization has to be a list of two integers");
    }
    return gko::precision_reduction(std::stoi(parts[0]), std::stoi(parts[1]));
}


const std::map<std::string, std::function<std::unique_ptr<gko::LinOpFactory>(
                                std::shared_ptr<const gko::Executor>)>>
    precond_factory{
        {"none",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::matrix::IdentityFactory<>::create(exec);
         }},
        {"jacobi",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::preconditioner::Jacobi<>::build()
                 .with_max_block_size(FLAGS_jacobi_max_block_size)
                 .with_storage_optimization(
                     parse_storage_optimization(FLAGS_jacobi_storage))
                 .with_accuracy(FLAGS_jacobi_accuracy)
                 .with_skip_sorting(true)
                 .on(exec);
         }},
        {"parict",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::ParIct<>::build()
                     .with_iterations(FLAGS_parilu_iterations)
                     .with_approximate_select(FLAGS_parilut_approx_select)
                     .with_fill_in_limit(FLAGS_parilut_limit)
                     .with_skip_sorting(true)
                     .on(exec));
             return gko::preconditioner::Ilu<>::build()
                 .with_factorization_factory(fact)
                 .on(exec);
         }},
        {"parilu",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::ParIlu<>::build()
                                .with_iterations(FLAGS_parilu_iterations)
                                .with_skip_sorting(true)
                                .on(exec));
             return gko::preconditioner::Ilu<>::build()
                 .with_factorization_factory(fact)
                 .on(exec);
         }},
        {"parilut",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::ParIlut<>::build()
                     .with_iterations(FLAGS_parilu_iterations)
                     .with_approximate_select(FLAGS_parilut_approx_select)
                     .with_fill_in_limit(FLAGS_parilut_limit)
                     .with_skip_sorting(true)
                     .on(exec));
             return gko::preconditioner::Ilu<>::build()
                 .with_factorization_factory(fact)
                 .on(exec);
         }},
        {"ilu",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::Ilu<>::build().on(exec));
             return gko::preconditioner::Ilu<>::build()
                 .with_factorization_factory(fact)
                 .on(exec);
         }},
        {"parict-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::ParIct<>::build()
                     .with_iterations(FLAGS_parilu_iterations)
                     .with_approximate_select(FLAGS_parilut_approx_select)
                     .with_fill_in_limit(FLAGS_parilut_limit)
                     .with_skip_sorting(true)
                     .on(exec));
             auto lisai = gko::share(gko::preconditioner::LowerIsai<>::build()
                                         .with_sparsity_power(FLAGS_isai_power)
                                         .on(exec));
             auto uisai = gko::share(gko::preconditioner::UpperIsai<>::build()
                                         .with_sparsity_power(FLAGS_isai_power)
                                         .on(exec));
             return gko::preconditioner::Ilu<
                        gko::preconditioner::LowerIsai<>,
                        gko::preconditioner::UpperIsai<>>::build()
                 .with_factorization_factory(fact)
                 .with_l_solver_factory(lisai)
                 .with_u_solver_factory(uisai)
                 .on(exec);
         }},
        {"parilu-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::ParIlu<>::build()
                                .with_iterations(FLAGS_parilu_iterations)
                                .with_skip_sorting(true)
                                .on(exec));
             auto lisai = gko::share(gko::preconditioner::LowerIsai<>::build()
                                         .with_sparsity_power(FLAGS_isai_power)
                                         .on(exec));
             auto uisai = gko::share(gko::preconditioner::UpperIsai<>::build()
                                         .with_sparsity_power(FLAGS_isai_power)
                                         .on(exec));
             return gko::preconditioner::Ilu<
                        gko::preconditioner::LowerIsai<>,
                        gko::preconditioner::UpperIsai<>>::build()
                 .with_factorization_factory(fact)
                 .with_l_solver_factory(lisai)
                 .with_u_solver_factory(uisai)
                 .on(exec);
         }},
        {"parilut-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::ParIlut<>::build()
                     .with_iterations(FLAGS_parilu_iterations)
                     .with_approximate_select(FLAGS_parilut_approx_select)
                     .with_fill_in_limit(FLAGS_parilut_limit)
                     .with_skip_sorting(true)
                     .on(exec));
             auto lisai = gko::share(gko::preconditioner::LowerIsai<>::build()
                                         .with_sparsity_power(FLAGS_isai_power)
                                         .on(exec));
             auto uisai = gko::share(gko::preconditioner::UpperIsai<>::build()
                                         .with_sparsity_power(FLAGS_isai_power)
                                         .on(exec));
             return gko::preconditioner::Ilu<
                        gko::preconditioner::LowerIsai<>,
                        gko::preconditioner::UpperIsai<>>::build()
                 .with_factorization_factory(fact)
                 .with_l_solver_factory(lisai)
                 .with_u_solver_factory(uisai)
                 .on(exec);
         }},
        {"ilu-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::Ilu<>::build().on(exec));
             auto lisai = gko::share(gko::preconditioner::LowerIsai<>::build()
                                         .with_sparsity_power(FLAGS_isai_power)
                                         .on(exec));
             auto uisai = gko::share(gko::preconditioner::UpperIsai<>::build()
                                         .with_sparsity_power(FLAGS_isai_power)
                                         .on(exec));
             return gko::preconditioner::Ilu<
                        gko::preconditioner::LowerIsai<>,
                        gko::preconditioner::UpperIsai<>>::build()
                 .with_factorization_factory(fact)
                 .with_l_solver_factory(lisai)
                 .with_u_solver_factory(uisai)
                 .on(exec);
         }},
        {"overhead", [](std::shared_ptr<const gko::Executor> exec) {
             return gko::Overhead<>::build().on(exec);
         }}};


#endif  // GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_
