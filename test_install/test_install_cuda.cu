/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


// core/base/polymorphic_object.hpp
class PolymorphicObjectTest : public gko::PolymorphicObject {};


int main(int, char **)
{
    auto refExec = gko::ReferenceExecutor::create();
    auto cudaExec = gko::CudaExecutor::create(0, refExec);
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
        using ArrayType = gko::Array<type1>;
        ArrayType{};
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
        gko::dim<3, type1>{4, 4, 4};
    }

    // core/base/exception.hpp
    {
        gko::Error(std::string("file"), 12,
                   std::string("Test for an error class."));
    }

    // core/base/exception_helpers.hpp
    {
        auto test = gko::dim<2>{3};
        GKO_ASSERT_IS_SQUARE_MATRIX(test);
    }

    // core/base/executor.hpp
    {
        gko::ReferenceExecutor::create();
    }

    // core/base/math.hpp
    {
        using testType = double;
        static_assert(gko::is_complex<testType>() == false,
                      "math.hpp not included properly!");
    }

    // core/base/matrix_data.hpp
    {
        gko::matrix_data<>{};
    }

    // core/base/mtx_io.hpp
    {
        static_assert(gko::layout_type::array != gko::layout_type::coordinate,
                      "mtx_io.hpp not included properly!");
    }

    // core/base/name_demangling.hpp
    {
        auto testVar = 3.0;
        gko::name_demangling::get_static_type(testVar);
    }


    // core/base/polymorphic_object.hpp
    {
        gko::PolymorphicObject *test;
        (void)test;  // silence unused variable warning
    }

    // core/base/range.hpp
    {
        gko::span{12};
    }

    // core/base/range_accessors.hpp
    {
        auto testVar = 12;
        gko::range<gko::accessor::row_major<decltype(testVar), 2>>(&testVar, 1u,
                                                                   1u, 1u);
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
                      "std_extensions.hpp not included properly!");
    }

    // core/base/types.hpp
    {
        static_assert(gko::size_type{12} == 12,
                      "types.hpp not included properly");
    }

    // core/base/utils.hpp
    {
        gko::null_deleter<double>{};
    }

    // core/base/version.hpp
    {
        gko::version_info::get().header_version;
    }

    // core/factorization/par_ilu.hpp
    {
        gko::factorization::ParIlu<>::build().on(cudaExec);
    }

    // core/log/convergence.hpp
    {
        gko::log::Convergence<>::create(cudaExec);
    }

    // core/log/record.hpp
    {
        gko::log::executor_data{};
    }

    // core/log/stream.hpp
    {
        gko::log::Stream<>::create(cudaExec);
    }

#if GKO_HAVE_PAPI_SDE
    // core/log/papi.hpp
    {
        gko::log::Papi<>::create(cudaExec);
    }
#endif  // GKO_HAVE_PAPI_SDE

    // core/matrix/coo.hpp
    {
        using Mtx = gko::matrix::Coo<>;
        Mtx::create(cudaExec, gko::dim<2>{2, 2}, 2);
    }

    // core/matrix/csr.hpp
    {
        using Mtx = gko::matrix::Csr<>;
        Mtx::create(cudaExec, gko::dim<2>{2, 2}, 2,
                    std::make_shared<Mtx::load_balance>(2));
    }

    // core/matrix/dense.hpp
    {
        using Mtx = gko::matrix::Dense<>;
        Mtx::create(cudaExec, gko::dim<2>{2, 2});
    }

    // core/matrix/ell.hpp
    {
        using Mtx = gko::matrix::Ell<>;
        Mtx::create(cudaExec, gko::dim<2>{2, 2}, 2);
    }

    // core/matrix/hybrid.hpp
    {
        using Mtx = gko::matrix::Hybrid<>;
        Mtx::create(cudaExec, gko::dim<2>{2, 2}, 2, 2, 1);
    }

    // core/matrix/identity.hpp
    {
        using Mtx = gko::matrix::Identity<>;
        Mtx::create(cudaExec);
    }

    // core/matrix/permutation.hpp
    {
        using Mtx = gko::matrix::Permutation<>;
        Mtx::create(cudaExec, gko::dim<2>{2, 2});
    }

    // core/matrix/sellp.hpp
    {
        using Mtx = gko::matrix::Sellp<>;
        Mtx::create(cudaExec, gko::dim<2>{2, 2}, 2);
    }

    // core/matrix/sparsity_csr.hpp
    {
        using Mtx = gko::matrix::SparsityCsr<>;
        Mtx::create(cudaExec, gko::dim<2>{2, 2});
    }

    // core/preconditioner/ilu.hpp
    {
        gko::preconditioner::Ilu<>::build().on(cudaExec);
    }

    // core/preconditioner/jacobi.hpp
    {
        using Bj = gko::preconditioner::Jacobi<>;
        Bj::build().with_max_block_size(1u).on(cudaExec);
    }

    // core/solver/bicgstab.hpp
    {
        using Solver = gko::solver::Bicgstab<>;
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(cudaExec))
            .on(cudaExec);
    }

    // core/solver/cg.hpp
    {
        using Solver = gko::solver::Cg<>;
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(cudaExec))
            .on(cudaExec);
    }

    // core/solver/cgs.hpp
    {
        using Solver = gko::solver::Cgs<>;
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(cudaExec))
            .on(cudaExec);
    }

    // core/solver/fcg.hpp
    {
        using Solver = gko::solver::Fcg<>;
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(cudaExec))
            .on(cudaExec);
    }

    // core/solver/gmres.hpp
    {
        using Solver = gko::solver::Gmres<>;
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(cudaExec))
            .on(cudaExec);
    }

    // core/solver/ir.hpp
    {
        using Solver = gko::solver::Ir<>;
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(cudaExec))
            .on(cudaExec);
    }

    // core/solver/lower_trs.hpp
    {
        using Solver = gko::solver::LowerTrs<>;
        Solver::build().on(cudaExec);
    }

    // core/stop/
    {
        // iteration.hpp
        auto iteration =
            gko::stop::Iteration::build().with_max_iters(1u).on(cudaExec);

        // time.hpp
        auto time = gko::stop::Time::build()
                        .with_time_limit(std::chrono::milliseconds(10))
                        .on(cudaExec);
        // residual_norm.hpp
        gko::stop::ResidualNormReduction<>::build()
            .with_reduction_factor(1e-10)
            .on(cudaExec);

        // stopping_status.hpp
        gko::stopping_status{};

        // combined.hpp
        auto combined =
            gko::stop::Combined::build()
                .with_criteria(std::move(time), std::move(iteration))
                .on(cudaExec);
    }

    std::cout
        << "test_install_cuda: the Ginkgo installation was correctly detected "
           "and is complete."
        << std::endl;

    return 0;
}
