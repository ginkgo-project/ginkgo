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

#include <ginkgo/core/matrix/diagonal.hpp>


#include <algorithm>
#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/diagonal_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Diagonal : public ::testing::Test {
protected:
    using value_type = T;
    using Diag = gko::matrix::Diagonal<value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    Diagonal()
        : exec(gko::ReferenceExecutor::create()),
          diag1(Diag::create(exec, 2)),
          diag2(Diag::create(exec, 3)),
          mtx1(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                    exec)),
          mtx2(
              gko::initialize<Mtx>({I<T>({1.0, 2.0}), I<T>({0.5, 1.5})}, exec)),
          mtx3(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                    exec)),
          mtx4(gko::initialize<Mtx>({I<T>({1.0, 2.0}), I<T>({0.5, 1.5})}, exec))
    {
        this->create_diag1(diag1.get());
        this->create_diag2(diag2.get());
    }

    void create_diag1(Diag *d)
    {
        auto *v = d->get_values();
        v[0] = 2.0;
        v[1] = 3.0;
    }

    void create_diag2(Diag *d)
    {
        auto *v = d->get_values();
        v[0] = 2.0;
        v[1] = 3.0;
        v[2] = 4.0;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Diag> diag1;
    std::unique_ptr<Diag> diag2;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    std::unique_ptr<Mtx> mtx4;
};

TYPED_TEST_CASE(Diagonal, gko::test::ValueTypes);


TYPED_TEST(Diagonal, AppliesToDense)
{
    using T = typename TestFixture::value_type;
    this->diag1->apply(this->mtx1.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx3->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx3->at(0, 1), T{4.0});
    EXPECT_EQ(this->mtx3->at(0, 2), T{6.0});
    EXPECT_EQ(this->mtx3->at(1, 0), T{4.5});
    EXPECT_EQ(this->mtx3->at(1, 1), T{7.5});
    EXPECT_EQ(this->mtx3->at(1, 2), T{10.5});
}


TYPED_TEST(Diagonal, RightAppliesToDense)
{
    using T = typename TestFixture::value_type;
    this->diag2->rapply(this->mtx1.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx3->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx3->at(0, 1), T{6.0});
    EXPECT_EQ(this->mtx3->at(0, 2), T{12.0});
    EXPECT_EQ(this->mtx3->at(1, 0), T{3.0});
    EXPECT_EQ(this->mtx3->at(1, 1), T{7.5});
    EXPECT_EQ(this->mtx3->at(1, 2), T{14.0});
}

}  // namespace
