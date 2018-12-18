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


int main(int argc, char *argv[])
{
    auto refExec = gko::ReferenceExecutor::create();
    // core/base/abstract_factory.hpp
    {
        using type1 = int;
        using type2 = double;
        static_assert(std::is_same<
                      gko::AbstractFactory<type1, type2>::abstract_product_type,
                      type1>::value);
    }

    // core/base/array.hpp
    {
        using type1 = int;
        using ArrayType = gko::Array<type1>;
        ArrayType test;
    }

    // core/base/combination.hpp
    {
        using type1 = int;
        static_assert(
            std::is_same<gko::Combination<type1>::value_type, type1>::value);
    }

    // core/base/composition.hpp
    {
        using type1 = int;
        static_assert(
            std::is_same<gko::Composition<type1>::value_type, type1>::value);
    }

    // core/base/dim.hpp
    {
        using type1 = int;
        auto test = gko::dim<3, type1>{4, 4, 4};
    }

    // core/base/exception.hpp
    {
        auto test = gko::Error(std::string("file"), 12,
                               std::string("Test for an error class."));
    }

    // core/base/exception_helpers.hpp
    {
        auto test = gko::dim<2>{3};
        ASSERT_IS_SQUARE_MATRIX(test);
    }

    // core/base/executor.hpp
    {
        auto test = gko::ReferenceExecutor::create();
    }

    // core/base/math.hpp
    {
        using testType = double;
        static_assert(gko::is_complex<testType>() == false);
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
            &testVar, 1, 1, 1);
    }

    // core/base/std_extensions.hpp
    {
        static_assert(std::is_same<gko::xstd::void_t<double>, void>::value);
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

    // core/log/convergence.hpp
    {
        auto test = gko::log::Convergence<>::create(refExec);
    }

    // core/log/record.hpp
    {
        auto test = gko::log::executor_data{};
    }

    // core/log/stream.hpp
    {
        auto test = gko::log::Stream<>::create(refExec);
    }

    // core/log/convergence.hpp
    {
        auto test = gko::log::Convergence<>::create(refExec);
    }

    // core/matrix/coo.hpp
    {
        using Mtx = gko::matrix::Coo<>;
        auto test = Mtx::create(refExec, gko::dim<2>{2, 2}, 2);
    }

    // core/matrix/csr.hpp
    {
        using Mtx = gko::matrix::Csr<>;
        auto test = Mtx::create(refExec, gko::dim<2>{2, 2}, 2,
                                std::make_shared<Mtx::load_balance>(2));
    }

    // core/matrix/dense.hpp
    {
        using Mtx = gko::matrix::Dense<>;
        auto test = Mtx::create(refExec, gko::dim<2>{2, 2});
    }

    // core/matrix/ell.hpp
    {
        using Mtx = gko::matrix::Ell<>;
        auto test = Mtx::create(refExec, gko::dim<2>{2, 2}, 2);
    }

    // core/matrix/hybrid.hpp
    {
        using Mtx = gko::matrix::Hybrid<>;
        auto test = Mtx::create(refExec, gko::dim<2>{2, 2}, 2, 2, 1);
    }

    // core/matrix/identity.hpp
    {
        using Mtx = gko::matrix::Identity<>;
        auto test = Mtx::create(refExec);
    }

    // core/matrix/sellp.hpp
    {
        using Mtx = gko::matrix::Sellp<>;
        auto test = Mtx::create(refExec, gko::dim<2>{2, 2}, 2);
    }

    // core/preconditioner/jacobi.hpp
    {
        using Bj = gko::preconditioner::Jacobi<>;
        auto test = Bj::build().with_max_block_size(1u).on(refExec);
    }

    // core/solver/bicgstab.hpp
    {
        using Solver = gko::solver::Bicgstab<>;
        auto test = Solver::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(1u).on(
                                refExec))
                        .on(refExec);
    }

    // core/solver/cg.hpp
    {
        using Solver = gko::solver::Cg<>;
        auto test = Solver::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(1u).on(
                                refExec))
                        .on(refExec);
    }

    // core/solver/cgs.hpp
    {
        using Solver = gko::solver::Cgs<>;
        auto test = Solver::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(1u).on(
                                refExec))
                        .on(refExec);
    }

    // core/solver/fcg.hpp
    {
        using Solver = gko::solver::Fcg<>;
        auto test = Solver::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(1u).on(
                                refExec))
                        .on(refExec);
    }

    // core/solver/gmres.hpp
    {
        using Solver = gko::solver::Gmres<>;
        auto test = Solver::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(1u).on(
                                refExec))
                        .on(refExec);
    }

    // core/stop/
    {
        // iteration.hpp
        auto iteration =
            gko::stop::Iteration::build().with_max_iters(1u).on(refExec);

        // time.hpp
        auto time = gko::stop::Time::build()
                        .with_time_limit(std::chrono::milliseconds(10))
                        .on(refExec);
        // residual_norm_reduction.hpp
        auto res_red = gko::stop::ResidualNormReduction<>::build()
                           .with_reduction_factor(1e-10)
                           .on(refExec);

        // stopping_status.hpp
        auto stop_status = gko::stopping_status{};

        // combined.hpp
        auto combined =
            gko::stop::Combined::build()
                .with_criteria(std::move(time), std::move(iteration))
                .on(refExec);
    }


    return 0;
}
