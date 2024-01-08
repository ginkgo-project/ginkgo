// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_
#define GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_


#include <ginkgo/ginkgo.hpp>


#include <map>
#include <string>


#include <gflags/gflags.h>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/overhead_linop.hpp"
#include "benchmark/utils/types.hpp"


DEFINE_string(preconditioners, "none",
              "A comma-separated list of preconditioners to use. "
              "Supported values are: none, jacobi, paric, parict, parilu, "
              "parilut, ic, ilu, paric-isai, parict-isai, parilu-isai, "
              "parilut-isai, ic-isai, ilu-isai, overhead");

DEFINE_uint32(parilu_iterations, 5,
              "The number of iterations for ParIC(T)/ParILU(T)");

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
gko::precision_reduction parse_storage_optimization(const std::string& flag)
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
             return gko::matrix::IdentityFactory<etype>::create(exec);
         }},
        {"jacobi",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::preconditioner::Jacobi<etype, itype>::build()
                 .with_max_block_size(FLAGS_jacobi_max_block_size)
                 .with_storage_optimization(
                     parse_storage_optimization(FLAGS_jacobi_storage))
                 .with_accuracy(static_cast<rc_etype>(FLAGS_jacobi_accuracy))
                 .with_skip_sorting(true)
                 .on(exec);
         }},
        {"paric",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::ParIc<etype, itype>::build()
                                .with_iterations(FLAGS_parilu_iterations)
                                .with_skip_sorting(true)
                                .on(exec));
             return gko::preconditioner::Ic<gko::solver::LowerTrs<etype, itype>,
                                            itype>::build()
                 .with_factorization(fact)
                 .on(exec);
         }},
        {"parict",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::ParIct<etype, itype>::build()
                     .with_iterations(FLAGS_parilu_iterations)
                     .with_approximate_select(FLAGS_parilut_approx_select)
                     .with_fill_in_limit(FLAGS_parilut_limit)
                     .with_skip_sorting(true)
                     .on(exec));
             return gko::preconditioner::
                 Ilu<gko::solver::LowerTrs<etype, itype>,
                     gko::solver::UpperTrs<etype, itype>, false, itype>::build()
                     .with_factorization(fact)
                     .on(exec);
         }},
        {"parilu",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::ParIlu<etype, itype>::build()
                                .with_iterations(FLAGS_parilu_iterations)
                                .with_skip_sorting(true)
                                .on(exec));
             return gko::preconditioner::
                 Ilu<gko::solver::LowerTrs<etype, itype>,
                     gko::solver::UpperTrs<etype, itype>, false, itype>::build()
                     .with_factorization(fact)
                     .on(exec);
         }},
        {"parilut",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::ParIlut<etype, itype>::build()
                     .with_iterations(FLAGS_parilu_iterations)
                     .with_approximate_select(FLAGS_parilut_approx_select)
                     .with_fill_in_limit(FLAGS_parilut_limit)
                     .with_skip_sorting(true)
                     .on(exec));
             return gko::preconditioner::
                 Ilu<gko::solver::LowerTrs<etype, itype>,
                     gko::solver::UpperTrs<etype, itype>, false, itype>::build()
                     .with_factorization(fact)
                     .on(exec);
         }},
        {"ic",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::Ic<etype, itype>::build().on(exec));
             return gko::preconditioner::Ic<gko::solver::LowerTrs<etype, itype>,
                                            itype>::build()
                 .with_factorization(fact)
                 .on(exec);
         }},
        {"ilu",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::Ilu<etype, itype>::build().on(exec));
             return gko::preconditioner::
                 Ilu<gko::solver::LowerTrs<etype, itype>,
                     gko::solver::UpperTrs<etype, itype>, false, itype>::build()
                     .with_factorization(fact)
                     .on(exec);
         }},
        {"paric-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::ParIc<etype, itype>::build()
                                .with_iterations(FLAGS_parilu_iterations)
                                .with_skip_sorting(true)
                                .on(exec));
             auto lisai = gko::share(
                 gko::preconditioner::LowerIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             return gko::preconditioner::Ic<
                        gko::preconditioner::LowerIsai<etype, itype>,
                        itype>::build()
                 .with_factorization(fact)
                 .with_l_solver(lisai)
                 .on(exec);
         }},
        {"parict-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::ParIct<etype, itype>::build()
                     .with_iterations(FLAGS_parilu_iterations)
                     .with_approximate_select(FLAGS_parilut_approx_select)
                     .with_fill_in_limit(FLAGS_parilut_limit)
                     .with_skip_sorting(true)
                     .on(exec));
             auto lisai = gko::share(
                 gko::preconditioner::LowerIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             return gko::preconditioner::Ic<
                        gko::preconditioner::LowerIsai<etype, itype>,
                        itype>::build()
                 .with_factorization(fact)
                 .with_l_solver(lisai)
                 .on(exec);
         }},
        {"parilu-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::ParIlu<etype, itype>::build()
                                .with_iterations(FLAGS_parilu_iterations)
                                .with_skip_sorting(true)
                                .on(exec));
             auto lisai = gko::share(
                 gko::preconditioner::LowerIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             auto uisai = gko::share(
                 gko::preconditioner::UpperIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             return gko::preconditioner::Ilu<
                        gko::preconditioner::LowerIsai<etype, itype>,
                        gko::preconditioner::UpperIsai<etype, itype>, false,
                        itype>::build()
                 .with_factorization(fact)
                 .with_l_solver(lisai)
                 .with_u_solver(uisai)
                 .on(exec);
         }},
        {"parilut-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::ParIlut<etype, itype>::build()
                     .with_iterations(FLAGS_parilu_iterations)
                     .with_approximate_select(FLAGS_parilut_approx_select)
                     .with_fill_in_limit(FLAGS_parilut_limit)
                     .with_skip_sorting(true)
                     .on(exec));
             auto lisai = gko::share(
                 gko::preconditioner::LowerIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             auto uisai = gko::share(
                 gko::preconditioner::UpperIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             return gko::preconditioner::Ilu<
                        gko::preconditioner::LowerIsai<etype, itype>,
                        gko::preconditioner::UpperIsai<etype, itype>, false,
                        itype>::build()
                 .with_factorization(fact)
                 .with_l_solver(lisai)
                 .with_u_solver(uisai)
                 .on(exec);
         }},
        {"ic-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::Ic<etype, itype>::build().on(exec));
             auto lisai = gko::share(
                 gko::preconditioner::LowerIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             return gko::preconditioner::Ic<
                        gko::preconditioner::LowerIsai<etype, itype>,
                        itype>::build()
                 .with_factorization(fact)
                 .with_l_solver(lisai)
                 .on(exec);
         }},
        {"ilu-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::factorization::Ilu<etype, itype>::build().on(exec));
             auto lisai = gko::share(
                 gko::preconditioner::LowerIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             auto uisai = gko::share(
                 gko::preconditioner::UpperIsai<etype, itype>::build()
                     .with_sparsity_power(FLAGS_isai_power)
                     .on(exec));
             return gko::preconditioner::Ilu<
                        gko::preconditioner::LowerIsai<etype, itype>,
                        gko::preconditioner::UpperIsai<etype, itype>, false,
                        itype>::build()
                 .with_factorization(fact)
                 .with_l_solver(lisai)
                 .with_u_solver(uisai)
                 .on(exec);
         }},
        {"general-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::preconditioner::GeneralIsai<etype, itype>::build()
                 .with_sparsity_power(FLAGS_isai_power)
                 .on(exec);
         }},
        {"spd-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::preconditioner::SpdIsai<etype, itype>::build()
                 .with_sparsity_power(FLAGS_isai_power)
                 .on(exec);
         }},
        {"overhead", [](std::shared_ptr<const gko::Executor> exec) {
             return gko::Overhead<etype>::build()
                 .with_criteria(gko::stop::ResidualNorm<etype>::build()
                                    .with_reduction_factor(rc_etype{}))
                 .on(exec);
         }}};


#endif  // GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_
