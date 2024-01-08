// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


void assert_similar_matrices(gko::ptr_param<const gko::matrix::Dense<>> m1,
                             gko::ptr_param<const gko::matrix::Dense<>> m2,
                             double prec)
{
    assert(m1->get_size()[0] == m2->get_size()[0]);
    assert(m1->get_size()[1] == m2->get_size()[1]);
    for (gko::size_type i = 0; i < m1->get_size()[0]; ++i) {
        for (gko::size_type j = 0; j < m2->get_size()[1]; ++j) {
            assert(std::abs(m1->at(i, j) - m2->at(i, j)) < prec);
        }
    }
}


template <typename Mtx>
void check_spmv(std::shared_ptr<gko::Executor> exec,
                const gko::matrix_data<double>& A_raw,
                gko::ptr_param<const gko::matrix::Dense<>> b,
                gko::ptr_param<gko::matrix::Dense<>> x)
{
    auto test = Mtx::create(exec);
#if HAS_REFERENCE
    auto x_clone = gko::clone(x);
    test->read(A_raw);
    test->apply(b, x_clone);
    // x_clone has the device result if using HIP or CUDA, otherwise it is
    // reference only

#if defined(HAS_HIP) || defined(HAS_CUDA)
    // If we are on a device, we need to run a reference test to compare against
    auto exec_ref = exec->get_master();
    auto test_ref = Mtx::create(exec_ref);
    auto x_ref = gko::clone(exec_ref, x);
    test_ref->read(A_raw);
    test_ref->apply(b, x_ref);

    // Actually check that `x_clone` is similar to `x_ref`
    auto x_clone_ref = gko::clone(exec_ref, x_clone);
    assert_similar_matrices(x_clone_ref, x_ref, 1e-14);
#endif  // defined(HAS_HIP) || defined(HAS_CUDA)
#endif  // HAS_REFERENCE
}


template <typename Solver>
void check_solver(std::shared_ptr<gko::Executor> exec,
                  const gko::matrix_data<double>& A_raw,
                  gko::ptr_param<const gko::matrix::Dense<>> b,
                  gko::ptr_param<gko::matrix::Dense<>> x)
{
    using Mtx = gko::matrix::Csr<>;
    auto A = gko::share(Mtx::create(exec, std::make_shared<Mtx::classical>()));

    auto num_iters = 20u;
    double reduction_factor = 1e-7;
    auto solver_gen =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(num_iters),
                gko::stop::ResidualNorm<>::build().with_reduction_factor(
                    reduction_factor))
            .on(exec);
#if HAS_REFERENCE
    A->read(A_raw);
    auto x_clone = gko::clone(x);
    solver_gen->generate(A)->apply(b, x_clone);
    // x_clone has the device result if using HIP or CUDA, otherwise it is
    // reference only

#if defined(HAS_HIP) || defined(HAS_CUDA)
    // If we are on a device, we need to run a reference test to compare against
    auto exec_ref = exec->get_master();
    auto A_ref =
        gko::share(Mtx::create(exec_ref, std::make_shared<Mtx::classical>()));
    A_ref->read(A_raw);
    auto solver_gen_ref =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(num_iters),
                gko::stop::ResidualNorm<>::build().with_reduction_factor(
                    reduction_factor))
            .on(exec_ref);
    auto x_ref = gko::clone(exec_ref, x);
    solver_gen->generate(A_ref)->apply(b, x_ref);

    // Actually check that `x_clone` is similar to `x_ref`
    auto x_clone_ref = gko::clone(exec_ref, x_clone);
    assert_similar_matrices(x_clone_ref, x_ref, 1e-12);
#endif  // defined(HAS_HIP) || defined(HAS_CUDA)
#endif  // HAS_REFERENCE
}


// core/base/polymorphic_object.hpp
class PolymorphicObjectTest : public gko::PolymorphicObject {};


int main()
{
#if defined(HAS_CUDA)
    auto extra_info = "(CUDA)";
    using exec_type = gko::CudaExecutor;
#elif defined(HAS_HIP)
    auto extra_info = "(HIP)";
    using exec_type = gko::HipExecutor;
#else
    auto extra_info = "(REFERENCE)";
    using exec_type = gko::ReferenceExecutor;
#endif

    std::shared_ptr<exec_type> exec;
    try {
#if defined(HAS_CUDA) || defined(HAS_HIP)
        exec = exec_type::create(0, gko::ReferenceExecutor::create());
#else
        exec = exec_type::create();
#endif
        // We also try to to synchronize to ensure we really have an available
        // device
        exec->synchronize();
    } catch (gko::Error& e) {
        // Exit gracefully to not trigger CI errors. We only skip the tests in
        // this setting
        std::cerr
            << "test_install" << extra_info
            << ": a compatible device could not be found. Skipping test.\n";
        std::exit(0);
    }

    using vec = gko::matrix::Dense<>;
#if HAS_REFERENCE
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);
    std::ifstream A_file("data/A.mtx");
    auto A_raw = gko::read_raw<double>(A_file);
#else
    // Instantiate dummy data, they will be unused
    auto b = vec::create(exec);
    auto x = vec::create(exec);
    gko::matrix_data<double> A_raw{};
#endif

    // core/base/abstract_factory.hpp
    {
        using type1 = int;
        using type2 = double;
        static_assert(
            std::is_same<
                gko::AbstractFactory<type1, type2>::abstract_product_type,
                type1>::value,
            "abstract_factory.hpp not included properly!");
    }

    // core/base/array.hpp
    {
        using type1 = int;
        using array_type = gko::array<type1>;
        array_type test;
    }

    // core/base/batch_dim.hpp
    {
        using type1 = int;
        auto test = gko::batch_dim<2, type1>{};
    }

    // core/base/batch_multi_vector.hpp
    {
        using type1 = float;
        using batch_multi_vector_type = gko::batch::MultiVector<type1>;
        auto test = batch_multi_vector_type::create(exec);
    }

    // core/base/combination.hpp
    {
        using type1 = int;
        static_assert(
            std::is_same<gko::Combination<type1>::value_type, type1>::value,
            "combination.hpp not included properly!");
    }

    // core/base/composition.hpp
    {
        using type1 = int;
        static_assert(
            std::is_same<gko::Composition<type1>::value_type, type1>::value,
            "composition.hpp not included properly");
    }

    // core/base/dim.hpp
    {
        using type1 = int;
        auto test = gko::dim<3, type1>{4, 4, 4};
    }

    // core/base/device_matrix_data.hpp
    {
        auto test =
            gko::device_matrix_data<float, int>{exec, gko::dim<2>{1, 1}, 1};
    }

    // core/base/exception.hpp
    {
        auto test = gko::Error(std::string("file"), 12,
                               std::string("Test for an error class."));
    }

    // core/base/exception_helpers.hpp
    {
        auto test = gko::dim<2>{3};
        GKO_ASSERT_IS_SQUARE_MATRIX(test);
    }

    // core/base/executor.hpp
    {
        auto test = gko::ReferenceExecutor::create();
    }

    // core/base/math.hpp
    {
        using testType = double;
        static_assert(gko::is_complex<testType>() == false,
                      "math.hpp not included properly!");
    }

    // core/base/matrix_data.hpp
    {
        gko::matrix_data<> test{};
    }

    // core/base/mtx_io.hpp
    {
        auto test = gko::layout_type::array;
    }

    // core/base/name_demangling.hpp
    {
        auto testVar = 3.0;
        auto test = gko::name_demangling::get_static_type(testVar);
    }

    // core/base/polymorphic_object.hpp
    {
        auto test = gko::layout_type::array;
    }

    // core/base/range.hpp
    {
        auto test = gko::span{12};
    }

    // core/base/range_accessors.hpp
    {
        auto testVar = 12;
        auto test = gko::range<gko::accessor::row_major<decltype(testVar), 2>>(
            &testVar, 1u, 1u, 1u);
    }

    // core/base/perturbation.hpp
    {
        using type1 = int;
        static_assert(
            std::is_same<gko::Perturbation<type1>::value_type, type1>::value,
            "perturbation.hpp not included properly");
    }

    // core/base/std_extensions.hpp
    {
        static_assert(std::is_same<gko::xstd::void_t<double>, void>::value,
                      "std::extensions.hpp not included properly!");
    }

    // core/base/types.hpp
    {
        gko::size_type test{12};
    }

    // core/base/utils.hpp
    {
        auto test = gko::null_deleter<double>{};
    }

    // core/base/version.hpp
    {
        auto test = gko::version_info::get().header_version;
    }

    // core/factorization/par_ilu.hpp
    {
        auto test = gko::factorization::ParIlu<>::build().on(exec);
    }

    // core/log/convergence.hpp
    {
        auto test = gko::log::Convergence<>::create();
    }

    // core/log/record.hpp
    {
        auto test = gko::log::executor_data{};
    }

    // core/log/stream.hpp
    {
        auto test = gko::log::Stream<>::create();
    }

#if GKO_HAVE_PAPI_SDE
    // core/log/papi.hpp
    {
        auto test = gko::log::Papi<>::create();
    }
#endif  // GKO_HAVE_PAPI_SDE

    // core/matrix/batch_dense.hpp
    {
        using type1 = float;
        using batch_dense_type = gko::batch::matrix::Dense<type1>;
        auto test = batch_dense_type::create(exec);
    }

    // core/matrix/batch_ell.hpp
    {
        using type1 = float;
        using batch_ell_type = gko::batch::matrix::Ell<type1>;
        auto test = batch_ell_type::create(exec);
    }

    // core/matrix/coo.hpp
    {
        using Mtx = gko::matrix::Coo<>;
        check_spmv<Mtx>(exec, A_raw, b, x);
    }

    // core/matrix/csr.hpp
    {
        using Mtx = gko::matrix::Csr<>;
        auto test = Mtx::create(exec, std::make_shared<Mtx::classical>());
    }

    // core/matrix/dense.hpp
    {
        using Mtx = gko::matrix::Dense<>;
        check_spmv<Mtx>(exec, A_raw, b, x);
    }

    // core/matrix/ell.hpp
    {
        using Mtx = gko::matrix::Ell<>;
        check_spmv<Mtx>(exec, A_raw, b, x);
    }

    // core/matrix/hybrid.hpp
    {
        using Mtx = gko::matrix::Hybrid<>;
        check_spmv<Mtx>(exec, A_raw, b, x);
    }

    // core/matrix/identity.hpp
    {
        using Mtx = gko::matrix::Identity<>;
        auto test = Mtx::create(exec);
    }

    // core/matrix/permutation.hpp
    {
        using Mtx = gko::matrix::Permutation<>;
        auto test = Mtx::create(exec, gko::dim<2>{2, 2});
    }

    // core/matrix/row_gatherer.hpp
    {
        using Mtx = gko::matrix::RowGatherer<>;
        auto test = Mtx::create(exec, gko::dim<2>{2, 2});
    }

    // core/matrix/sellp.hpp
    {
        using Mtx = gko::matrix::Sellp<>;
        check_spmv<Mtx>(exec, A_raw, b, x);
    }

    // core/matrix/sparsity_csr.hpp
    {
        using Mtx = gko::matrix::SparsityCsr<>;
        auto test = Mtx::create(exec, gko::dim<2>{2, 2});
    }

    // core/multigrid/pgm.hpp
    {
        auto test = gko::multigrid::Pgm<>::build().on(exec);
    }

    // core/preconditioner/ilu.hpp
    {
        auto test = gko::preconditioner::Ilu<>::build().on(exec);
    }

    // core/preconditioner/isai.hpp
    {
        auto test_l = gko::preconditioner::LowerIsai<>::build().on(exec);
        auto test_u = gko::preconditioner::UpperIsai<>::build().on(exec);
    }

    // core/preconditioner/jacobi.hpp
    {
        using Bj = gko::preconditioner::Jacobi<>;
        auto test = Bj::build().with_max_block_size(1u).on(exec);
    }

    // core/solver/batch_bicgstab.hpp
    {
        using Solver = gko::batch::solver::Bicgstab<>;
        auto test = Solver::build().with_max_iterations(5).on(exec);
    }

    // core/solver/bicgstab.hpp
    {
        using Solver = gko::solver::Bicgstab<>;
        check_solver<Solver>(exec, A_raw, b, x);
    }

    // core/solver/cb_gmres.hpp
    {
        using Solver = gko::solver::CbGmres<>;
        check_solver<Solver>(exec, A_raw, b, x);
    }

    // core/solver/cg.hpp
    {
        using Solver = gko::solver::Cg<>;
        check_solver<Solver>(exec, A_raw, b, x);
    }

    // core/solver/cgs.hpp
    {
        using Solver = gko::solver::Cgs<>;
        check_solver<Solver>(exec, A_raw, b, x);
    }

    // core/solver/fcg.hpp
    {
        using Solver = gko::solver::Fcg<>;
        check_solver<Solver>(exec, A_raw, b, x);
    }

    // core/solver/gcr.hpp
    {
        using Solver = gko::solver::Gcr<>;
        check_solver<Solver>(exec, A_raw, b, x);
    }

    // core/solver/gmres.hpp
    {
        using Solver = gko::solver::Gmres<>;
        check_solver<Solver>(exec, A_raw, b, x);
    }

    // core/solver/ir.hpp
    {
        using Solver = gko::solver::Ir<>;
        auto test =
            Solver::build()
                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                .on(exec);
    }

    // core/solver/lower_trs.hpp
    {
        using Solver = gko::solver::LowerTrs<>;
        auto test = Solver::build().on(exec);
    }

    // core/stop/
    {
        // iteration.hpp
        auto iteration =
            gko::stop::Iteration::build().with_max_iters(1u).on(exec);

        // time.hpp
        auto time = gko::stop::Time::build()
                        .with_time_limit(std::chrono::milliseconds(10))
                        .on(exec);

        // residual_norm.hpp
        auto main_res = gko::stop::ResidualNorm<>::build()
                            .with_reduction_factor(1e-10)
                            .with_baseline(gko::stop::mode::absolute)
                            .on(exec);

        auto implicit_res = gko::stop::ImplicitResidualNorm<>::build()
                                .with_reduction_factor(1e-10)
                                .with_baseline(gko::stop::mode::absolute)
                                .on(exec);

        // stopping_status.hpp
        auto stop_status = gko::stopping_status{};

        // combined.hpp
        auto combined =
            gko::stop::Combined::build()
                .with_criteria(std::move(time), std::move(iteration))
                .on(exec);
    }
    std::cout << "test_install" << extra_info
              << ": the Ginkgo installation was correctly detected "
                 "and is complete."
              << std::endl;

    return 0;
}
