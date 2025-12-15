// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_
#define GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_


#include <map>
#include <string>

#include <gflags/gflags.h>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/overhead_linop.hpp"
#include "benchmark/utils/types.hpp"


// MSVC has different way to expand macro than linux, so we can not put the #if
// inside the DEFINE_string macro
#define PRECONDITIONERS_COMMON                                        \
    "A comma-separated list of preconditioners to use. "              \
    "Supported values are: none, jacobi, mg, paric, parict, parilu, " \
    "parilut, ic, ilu, paric-isai, parict-isai, parilu-isai, "        \
    "parilut-isai, ic-isai, ilu-isai, sor, overhead"
#if GINKGO_BUILD_MPI
DEFINE_string(preconditioners, "none",
              PRECONDITIONERS_COMMON
              ", schwarz-jacobi, schwarz-ilu, schwarz-ic, schwarz-lu");
#else
DEFINE_string(preconditioners, "none", PRECONDITIONERS_COMMON);
#endif

#undef PRECONDITIONERS_COMMON

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

DEFINE_double(sor_relaxation_factor, 1.0,
              "The relaxation factor for the SOR preconditioner");

DEFINE_bool(sor_symmetric, false,
            "Apply the SOR preconditioner symmetrically, i.e. use SSOR");

DEFINE_bool(pgm_deterministic, false,
            "Use deterministic computation of the aggregated group within PGM");

DEFINE_uint32(
    mg_max_num_levels, false,
    "The maximum number of levels to use for the Multigrid preconditioner");

DEFINE_double(mg_tolerance, false, "The tolerance for the coarse solver");

DEFINE_uint32(mg_max_iters, false,
              "The max number of iterations for the coarse solver");


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
        {"sor",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::preconditioner::Sor<etype, itype>::build()
                 .with_relaxation_factor(
                     static_cast<gko::remove_complex<etype>>(
                         FLAGS_sor_relaxation_factor))
                 .with_symmetric(FLAGS_sor_symmetric)
                 .on(exec);
         }},
        {"overhead",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::Overhead<etype>::build()
                 .with_criteria(gko::stop::ResidualNorm<etype>::build()
                                    .with_reduction_factor(rc_etype{}))
                 .on(exec);
         }},
        {"mg",
         [](std::shared_ptr<const gko::Executor> exec) {
             using ir = gko::solver::Ir<etype>;
             auto iter_stop = gko::share(gko::stop::Iteration::build()
                                             .with_max_iters(FLAGS_mg_max_iters)
                                             .on(exec));
             auto tol_stop =
                 gko::share(gko::stop::ResidualNorm<etype>::build()
                                .with_baseline(gko::stop::mode::absolute)
                                .with_reduction_factor(FLAGS_mg_tolerance)
                                .on(exec));
             return gko::solver::Multigrid::build()
                 .with_mg_level(
                     gko::multigrid::Pgm<etype, itype>::build()
                         .with_deterministic(FLAGS_pgm_deterministic))
                 .with_criteria(iter_stop, tol_stop)
                 .with_max_levels(FLAGS_mg_max_num_levels)
                 .on(exec);
         }}
#if GINKGO_BUILD_MPI
        ,
        {"schwarz-jacobi",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::experimental::distributed::preconditioner::Schwarz<
                        etype>::build()
                 .with_local_solver(
                     gko::preconditioner::Jacobi<etype>::build()
                         .with_max_block_size(FLAGS_jacobi_max_block_size)
                         .with_storage_optimization(
                             parse_storage_optimization(FLAGS_jacobi_storage))
                         .with_accuracy(
                             static_cast<rc_etype>(FLAGS_jacobi_accuracy))
                         .with_skip_sorting(true)
                         .on(exec))
                 .on(exec);
         }},
        {"schwarz-general-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::experimental::distributed::preconditioner::Schwarz<
                        etype, itype>::build()
                 .with_local_solver(
                     gko::preconditioner::GeneralIsai<etype, itype>::build()
                         .with_sparsity_power(FLAGS_isai_power)
                         .on(exec))
                 .on(exec);
         }},
        {"schwarz-spd-isai",
         [](std::shared_ptr<const gko::Executor> exec) {
             return gko::experimental::distributed::preconditioner::Schwarz<
                        etype, itype>::build()
                 .with_local_solver(
                     gko::preconditioner::SpdIsai<etype, itype>::build()
                         .with_sparsity_power(FLAGS_isai_power)
                         .on(exec))
                 .on(exec);
         }},
        {"schwarz-ilu",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::Ilu<etype, itype>::build()
                                .with_skip_sorting(true)
                                .on(exec));
             return gko::experimental::distributed::preconditioner::Schwarz<
                        etype, itype>::build()
                 .with_local_solver(gko::preconditioner::Ilu<
                                        gko::solver::LowerTrs<etype, itype>,
                                        gko::solver::UpperTrs<etype, itype>,
                                        false, itype>::build()
                                        .with_factorization(fact)
                                        .on(exec))
                 .on(exec);
         }},
        {"schwarz-ic",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact =
                 gko::share(gko::factorization::Ic<etype, itype>::build()
                                .with_skip_sorting(true)
                                .on(exec));
             return gko::experimental::distributed::preconditioner::Schwarz<
                        etype, itype>::build()
                 .with_local_solver(
                     gko::preconditioner::Ic<
                         gko::solver::LowerTrs<etype, itype>, itype>::build()
                         .with_factorization(fact)
                         .on(exec))
                 .on(exec);
         }},
        {"schwarz-lu",
         [](std::shared_ptr<const gko::Executor> exec) {
             auto fact = gko::share(
                 gko::experimental::factorization::Lu<etype, itype>::build().on(
                     exec));
             return gko::experimental::distributed::preconditioner::Schwarz<
                        etype, itype>::build()
                 .with_local_solver(
                     gko::experimental::solver::Direct<etype, itype>::build()
                         .with_factorization(fact)
                         .on(exec))
                 .on(exec);
         }}
#endif
    };


#endif  // GKO_BENCHMARK_UTILS_PRECONDITIONERS_HPP_
