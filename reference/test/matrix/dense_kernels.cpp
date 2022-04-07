/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/dense.hpp>


#include <complex>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class Dense : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using MixedMtx = gko::matrix::Dense<gko::next_precision<value_type>>;
    using ComplexMtx = gko::to_complex<Mtx>;
    using MixedComplexMtx = gko::to_complex<MixedMtx>;
    using RealMtx = gko::remove_complex<Mtx>;
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                    exec)),
          mtx2(gko::initialize<Mtx>({I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})},
                                    exec)),
          mtx3(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}},
                                    exec)),
          mtx4(gko::initialize<Mtx>(4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}},
                                    exec)),
          mtx5(gko::initialize<Mtx>(
              {{1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}, exec)),
          mtx6(gko::initialize<Mtx>({{1.0, 2.0, 0.0}, {0.0, 1.5, 0.0}}, exec)),
          mtx7(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 1.5, 0.0}}, exec)),
          mtx8(gko::initialize<Mtx>(
              {I<T>({1.0, -1.0}), I<T>({-2.0, 2.0}), I<T>({-3.0, 3.0})}, exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    std::unique_ptr<Mtx> mtx4;
    std::unique_ptr<Mtx> mtx5;
    std::unique_ptr<Mtx> mtx6;
    std::unique_ptr<Mtx> mtx7;
    std::unique_ptr<Mtx> mtx8;
    gko::int32 invalid_index = gko::invalid_index<gko::int32>();
    std::default_random_engine rand_engine;

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<gko::size_type>(num_cols, num_cols),
            std::normal_distribution<gko::remove_complex<value_type>>(0.0, 1.0),
            rand_engine, exec);
    }
};


TYPED_TEST_SUITE(Dense, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Dense, CopyRespectsStride)
{
    using value_type = typename TestFixture::value_type;
    auto m =
        gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, this->exec);
    auto m2 =
        gko::matrix::Dense<TypeParam>::create(this->exec, gko::dim<2>{2, 1}, 2);
    auto original_data = m2->get_values();
    original_data[1] = TypeParam{3.0};

    m->convert_to(m2.get());

    EXPECT_EQ(m2->at(0, 0), value_type{1.0});
    EXPECT_EQ(m2->get_stride(), 2);
    EXPECT_EQ(m2->at(1, 0), value_type{2.0});
    EXPECT_EQ(m2->get_values(), original_data);
    EXPECT_EQ(original_data[1], TypeParam{3.0});
}


TYPED_TEST(Dense, TemporaryOutputCloneWorks)
{
    using value_type = typename TestFixture::value_type;
    auto other = gko::OmpExecutor::create();
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, other);

    {
        auto clone = gko::make_temporary_output_clone(this->exec, m.get());
        clone->at(0) = 4.0;
        clone->at(1) = 5.0;

        ASSERT_EQ(m->at(0), value_type{1.0});
        ASSERT_EQ(m->at(1), value_type{2.0});
        ASSERT_EQ(clone->get_size(), m->get_size());
        ASSERT_EQ(clone->get_executor(), this->exec);
    }
    ASSERT_EQ(m->at(0), value_type{4.0});
    ASSERT_EQ(m->at(1), value_type{5.0});
}


TYPED_TEST(Dense, CanBeFilledWithValue)
{
    using value_type = typename TestFixture::value_type;
    auto m =
        gko::initialize<gko::matrix::Dense<TypeParam>>({1.0, 2.0}, this->exec);
    EXPECT_EQ(m->at(0), value_type{1});
    EXPECT_EQ(m->at(1), value_type{2});

    m->fill(value_type{42});

    EXPECT_EQ(m->at(0), value_type{42});
    EXPECT_EQ(m->at(1), value_type{42});
}


TYPED_TEST(Dense, CanBeFilledWithValueForStridedMatrices)
{
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto m = gko::initialize<gko::matrix::Dense<TypeParam>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, this->exec);
    T in_stride{-1.0};
    m->get_values()[3] = in_stride;

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), value_type{1.0});
    EXPECT_EQ(m->at(1), value_type{2.0});
    EXPECT_EQ(m->at(2), value_type{3.0});
    EXPECT_EQ(m->at(3), value_type{4.0});
    EXPECT_EQ(m->at(4), value_type{5.0});
    EXPECT_EQ(m->at(5), value_type{6.0});

    m->fill(value_type{42});

    ASSERT_EQ(m->get_size(), gko::dim<2>(3, 2));
    EXPECT_EQ(m->get_num_stored_elements(), 12);
    EXPECT_EQ(m->at(0), value_type{42.0});
    EXPECT_EQ(m->at(1), value_type{42.0});
    EXPECT_EQ(m->at(2), value_type{42.0});
    EXPECT_EQ(m->at(3), value_type{42.0});
    EXPECT_EQ(m->at(4), value_type{42.0});
    EXPECT_EQ(m->at(5), value_type{42.0});
    ASSERT_EQ(m->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesToDense)
{
    using T = typename TestFixture::value_type;
    T in_stride{-1};
    this->mtx3->get_values()[3] = in_stride;

    this->mtx2->apply(this->mtx1.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx3->at(0, 0), T{-0.5});
    EXPECT_EQ(this->mtx3->at(0, 1), T{-0.5});
    EXPECT_EQ(this->mtx3->at(0, 2), T{-0.5});
    EXPECT_EQ(this->mtx3->at(1, 0), T{1.0});
    EXPECT_EQ(this->mtx3->at(1, 1), T{1.0});
    EXPECT_EQ(this->mtx3->at(1, 2), T{1.0});
    ASSERT_EQ(this->mtx3->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesToMixedDense)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    auto mmtx1 = MixedMtx::create(this->exec);
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx1->convert_to(mmtx1.get());
    this->mtx3->convert_to(mmtx3.get());

    this->mtx2->apply(mmtx1.get(), mmtx3.get());

    EXPECT_EQ(mmtx3->at(0, 0), MixedT{-0.5});
    EXPECT_EQ(mmtx3->at(0, 1), MixedT{-0.5});
    EXPECT_EQ(mmtx3->at(0, 2), MixedT{-0.5});
    EXPECT_EQ(mmtx3->at(1, 0), MixedT{1.0});
    EXPECT_EQ(mmtx3->at(1, 1), MixedT{1.0});
    ASSERT_EQ(mmtx3->at(1, 2), MixedT{1.0});
}


TYPED_TEST(Dense, AppliesLinearCombinationToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({-1.0}, this->exec);
    auto beta = gko::initialize<Mtx>({2.0}, this->exec);
    T in_stride{-1};
    this->mtx3->get_values()[3] = in_stride;

    this->mtx2->apply(alpha.get(), this->mtx1.get(), beta.get(),
                      this->mtx3.get());

    EXPECT_EQ(this->mtx3->at(0, 0), T{2.5});
    EXPECT_EQ(this->mtx3->at(0, 1), T{4.5});
    EXPECT_EQ(this->mtx3->at(0, 2), T{6.5});
    EXPECT_EQ(this->mtx3->at(1, 0), T{0.0});
    EXPECT_EQ(this->mtx3->at(1, 1), T{2.0});
    EXPECT_EQ(this->mtx3->at(1, 2), T{4.0});
    ASSERT_EQ(this->mtx3->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AppliesLinearCombinationToMixedDense)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    auto mmtx1 = MixedMtx::create(this->exec);
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx1->convert_to(mmtx1.get());
    this->mtx3->convert_to(mmtx3.get());
    auto alpha = gko::initialize<MixedMtx>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedMtx>({2.0}, this->exec);

    this->mtx2->apply(alpha.get(), mmtx1.get(), beta.get(), mmtx3.get());

    EXPECT_EQ(mmtx3->at(0, 0), MixedT{2.5});
    EXPECT_EQ(mmtx3->at(0, 1), MixedT{4.5});
    EXPECT_EQ(mmtx3->at(0, 2), MixedT{6.5});
    EXPECT_EQ(mmtx3->at(1, 0), MixedT{0.0});
    EXPECT_EQ(mmtx3->at(1, 1), MixedT{2.0});
    ASSERT_EQ(mmtx3->at(1, 2), MixedT{4.0});
}


TYPED_TEST(Dense, ApplyFailsOnWrongInnerDimension)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(this->mtx1.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ApplyFailsOnWrongNumberOfRows)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{3});

    ASSERT_THROW(this->mtx1->apply(this->mtx2.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ApplyFailsOnWrongNumberOfCols)
{
    using Mtx = typename TestFixture::Mtx;
    auto res = Mtx::create(this->exec, gko::dim<2>{2}, 3);

    ASSERT_THROW(this->mtx1->apply(this->mtx2.get(), res.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({I<T>{2.0, -2.0}}, this->exec);

    this->mtx2->scale(alpha.get());

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{-4.0});
}


TYPED_TEST(Dense, ScalesDataMixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<MixedMtx>({I<MixedT>{2.0, -2.0}}, this->exec);

    this->mtx2->scale(alpha.get());

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{-4.0});
}


TYPED_TEST(Dense, InvScalesData)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({I<T>{0.5, -0.5}}, this->exec);

    this->mtx2->inv_scale(alpha.get());

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{-4.0});
}


TYPED_TEST(Dense, ScalesDataWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);

    this->mtx2->scale(alpha.get());

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{-2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{4.0});
}


TYPED_TEST(Dense, InvScalesDataWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({0.5}, this->exec);

    this->mtx2->inv_scale(alpha.get());

    EXPECT_EQ(this->mtx2->at(0, 0), T{2.0});
    EXPECT_EQ(this->mtx2->at(0, 1), T{-2.0});
    EXPECT_EQ(this->mtx2->at(1, 0), T{-4.0});
    EXPECT_EQ(this->mtx2->at(1, 1), T{4.0});
}


TYPED_TEST(Dense, ScalesDataWithStride)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{-1.0, 1.0, 2.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->scale(alpha.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{-1.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{6.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{-1.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 2), T{7.0});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AddsScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{2.0, 1.0, -2.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->add_scaled(alpha.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{-3.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(1, 2), T{-1.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AddsScaledMixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx3->convert_to(mmtx3.get());
    auto alpha = gko::initialize<MixedMtx>({{2.0, 1.0, -2.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->add_scaled(alpha.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{-3.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(1, 2), T{-1.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AddsScale)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{2.0, 1.0, -2.0}}, this->exec);
    auto beta = gko::initialize<Mtx>({{-1.0, 2.0, -3.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->add_scale(alpha.get(), this->mtx3.get(), beta.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{1.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{6.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{-15.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{-0.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{6.5});
    EXPECT_EQ(this->mtx1->at(1, 2), T{-15.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AddScaleViaApply)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto beta = gko::initialize<Mtx>({-1.0}, this->exec);
    auto id = gko::matrix::Identity<T>::create(this->exec, gko::dim<2>{3, 3});
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx3->apply(alpha.get(), id.get(), beta.get(), this->mtx1.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{1.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{2.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{3.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{-0.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{0.5});
    EXPECT_EQ(this->mtx1->at(1, 2), T{1.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(Dense, SubtractsScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({{-2.0, -1.0, 2.0}}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->sub_scaled(alpha.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{-3.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{4.0});
    EXPECT_EQ(this->mtx1->at(1, 2), T{-1.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AddsScaledWithScalar)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    T in_stride{-1};
    this->mtx1->get_values()[3] = in_stride;

    this->mtx1->add_scaled(alpha.get(), this->mtx3.get());

    EXPECT_EQ(this->mtx1->at(0, 0), T{3.0});
    EXPECT_EQ(this->mtx1->at(0, 1), T{6.0});
    EXPECT_EQ(this->mtx1->at(0, 2), T{9.0});
    EXPECT_EQ(this->mtx1->at(1, 0), T{2.5});
    EXPECT_EQ(this->mtx1->at(1, 1), T{5.5});
    EXPECT_EQ(this->mtx1->at(1, 2), T{8.5});
    ASSERT_EQ(this->mtx1->get_values()[3], in_stride);
}


TYPED_TEST(Dense, AddScaledFailsOnWrongSizes)
{
    using Mtx = typename TestFixture::Mtx;
    auto alpha = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->add_scaled(alpha.get(), this->mtx2.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, AddsScaledDiag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({2.0}, this->exec);
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2, I<T>{3.0, 2.0});

    this->mtx2->add_scaled(alpha.get(), diag.get());

    ASSERT_EQ(this->mtx2->at(0, 0), T{7.0});
    ASSERT_EQ(this->mtx2->at(0, 1), T{-1.0});
    ASSERT_EQ(this->mtx2->at(1, 0), T{-2.0});
    ASSERT_EQ(this->mtx2->at(1, 1), T{6.0});
}


TYPED_TEST(Dense, SubtractsScaledDiag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({-2.0}, this->exec);
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2, I<T>{3.0, 2.0});

    this->mtx2->sub_scaled(alpha.get(), diag.get());

    ASSERT_EQ(this->mtx2->at(0, 0), T{7.0});
    ASSERT_EQ(this->mtx2->at(0, 1), T{-1.0});
    ASSERT_EQ(this->mtx2->at(1, 0), T{-2.0});
    ASSERT_EQ(this->mtx2->at(1, 1), T{6.0});
}


TYPED_TEST(Dense, ComputesDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_dot(this->mtx3.get(), result.get());

    EXPECT_EQ(result->at(0, 0), T{1.75});
    EXPECT_EQ(result->at(0, 1), T{7.75});
    ASSERT_EQ(result->at(0, 2), T{17.75});
}


TYPED_TEST(Dense, ComputesDotMixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx3->convert_to(mmtx3.get());
    auto result = MixedMtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_dot(this->mtx3.get(), result.get());

    EXPECT_EQ(result->at(0, 0), MixedT{1.75});
    EXPECT_EQ(result->at(0, 1), MixedT{7.75});
    ASSERT_EQ(result->at(0, 2), MixedT{17.75});
}


TYPED_TEST(Dense, ComputesConjDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_conj_dot(this->mtx3.get(), result.get());

    EXPECT_EQ(result->at(0, 0), T{1.75});
    EXPECT_EQ(result->at(0, 1), T{7.75});
    ASSERT_EQ(result->at(0, 2), T{17.75});
}


TYPED_TEST(Dense, ComputesConjDotMixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    auto mmtx3 = MixedMtx::create(this->exec);
    this->mtx3->convert_to(mmtx3.get());
    auto result = MixedMtx::create(this->exec, gko::dim<2>{1, 3});

    this->mtx1->compute_conj_dot(this->mtx3.get(), result.get());

    EXPECT_EQ(result->at(0, 0), MixedT{1.75});
    EXPECT_EQ(result->at(0, 1), MixedT{7.75});
    ASSERT_EQ(result->at(0, 2), MixedT{17.75});
}


TYPED_TEST(Dense, ComputesNorm2)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm2(result.get());

    EXPECT_EQ(result->at(0, 0), T_nc{3.0});
    EXPECT_EQ(result->at(0, 1), T_nc{5.0});
}


TYPED_TEST(Dense, ComputesNorm2Mixed)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    using MixedT_nc = gko::remove_complex<MixedT>;
    using MixedNormVector = gko::matrix::Dense<MixedT_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = MixedNormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm2(result.get());

    EXPECT_EQ(result->at(0, 0), MixedT_nc{3.0});
    EXPECT_EQ(result->at(0, 1), MixedT_nc{5.0});
}


TYPED_TEST(Dense, ComputesNorm2Squared)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    gko::array<char> tmp{this->exec};
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}}, this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    gko::kernels::reference::dense::compute_squared_norm2(
        gko::as<gko::ReferenceExecutor>(this->exec), mtx.get(), result.get(),
        tmp);

    EXPECT_EQ(result->at(0, 0), T_nc{9.0});
    EXPECT_EQ(result->at(0, 1), T_nc{25.0});
}


TYPED_TEST(Dense, ComputesSqrt)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    auto mtx(gko::initialize<NormVector>(I<I<T_nc>>{{9.0, 25.0}}, this->exec));

    gko::kernels::reference::dense::compute_sqrt(
        gko::as<gko::ReferenceExecutor>(this->exec), mtx.get());

    EXPECT_EQ(mtx->at(0, 0), T_nc{3.0});
    EXPECT_EQ(mtx->at(0, 1), T_nc{5.0});
}


TYPED_TEST(Dense, ComputesNorm1)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<T>;
    using NormVector = gko::matrix::Dense<T_nc>;
    auto mtx(gko::initialize<Mtx>(
        {I<T>{1.0, 0.0}, I<T>{2.0, 3.0}, I<T>{2.0, 4.0}, I<T>{-1.0, -1.0}},
        this->exec));
    auto result = NormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm1(result.get());

    EXPECT_EQ(result->at(0, 0), T_nc{6.0});
    EXPECT_EQ(result->at(0, 1), T_nc{8.0});
}


TYPED_TEST(Dense, ComputesNorm1Mixed)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using MixedT = typename MixedMtx::value_type;
    using MixedT_nc = gko::remove_complex<MixedT>;
    using MixedNormVector = gko::matrix::Dense<MixedT_nc>;
    auto mtx(
        gko::initialize<MixedMtx>({I<MixedT>{1.0, 0.0}, I<MixedT>{2.0, 3.0},
                                   I<MixedT>{2.0, 4.0}, I<MixedT>{-1.0, -1.0}},
                                  this->exec));
    auto result = MixedNormVector::create(this->exec, gko::dim<2>{1, 2});

    mtx->compute_norm1(result.get());

    EXPECT_EQ(result->at(0, 0), MixedT_nc{6.0});
    EXPECT_EQ(result->at(0, 1), MixedT_nc{8.0});
}


TYPED_TEST(Dense, ComputeDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    ASSERT_THROW(this->mtx1->compute_dot(this->mtx2.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ComputeDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->compute_dot(this->mtx3.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ComputeConjDotFailsOnWrongInputSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 3});

    ASSERT_THROW(this->mtx1->compute_conj_dot(this->mtx2.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ComputeConjDotFailsOnWrongResultSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto result = Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->mtx1->compute_conj_dot(this->mtx3.get(), result.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ConvertsToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto tmp = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx1->convert_to(tmp.get());
    tmp->convert_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Dense, MovesToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto tmp = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);
    // If OtherT is more precise: 0, otherwise r
    auto residual = r<OtherT>::value < r<T>::value
                        ? gko::remove_complex<T>{0}
                        : gko::remove_complex<T>{r<OtherT>::value};

    this->mtx1->move_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Dense, ConvertsToCoo32)
{
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int32>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->convert_to(coo_mtx.get());
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToCoo32)
{
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int32>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->move_to(coo_mtx.get());
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToCoo64)
{
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int64>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->convert_to(coo_mtx.get());
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToCoo64)
{
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int64>;
    auto coo_mtx = Coo::create(this->mtx4->get_executor());

    this->mtx4->move_to(coo_mtx.get());
    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToCsr32)
{
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int32>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);

    this->mtx4->convert_to(csr_mtx_c.get());
    this->mtx4->convert_to(csr_mtx_m.get());

    auto v = csr_mtx_c->get_const_values();
    auto c = csr_mtx_c->get_const_col_idxs();
    auto r = csr_mtx_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c.get(), csr_mtx_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Dense, MovesToCsr32)
{
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int32>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx4->clone();

    this->mtx4->move_to(csr_mtx_c.get());
    mtx_clone->move_to(csr_mtx_m.get());

    auto v = csr_mtx_c->get_const_values();
    auto c = csr_mtx_c->get_const_col_idxs();
    auto r = csr_mtx_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c.get(), csr_mtx_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Dense, ConvertsToCsr64)
{
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int64>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);

    this->mtx4->convert_to(csr_mtx_c.get());
    this->mtx4->convert_to(csr_mtx_m.get());

    auto v = csr_mtx_c->get_const_values();
    auto c = csr_mtx_c->get_const_col_idxs();
    auto r = csr_mtx_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c.get(), csr_mtx_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Dense, MovesToCsr64)
{
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int64>;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx4->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx4->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx4->clone();

    this->mtx4->move_to(csr_mtx_c.get());
    mtx_clone->move_to(csr_mtx_m.get());

    auto v = csr_mtx_c->get_const_values();
    auto c = csr_mtx_c->get_const_col_idxs();
    auto r = csr_mtx_c->get_const_row_ptrs();
    ASSERT_EQ(csr_mtx_c->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(csr_mtx_c->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    GKO_ASSERT_MTX_NEAR(csr_mtx_c.get(), csr_mtx_m.get(), 0.0);
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Dense, ConvertsToSparsityCsr32)
{
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int32>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->convert_to(sparsity_csr_mtx.get());
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
}


TYPED_TEST(Dense, MovesToSparsityCsr32)
{
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int32>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->move_to(sparsity_csr_mtx.get());
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
}


TYPED_TEST(Dense, ConvertsToSparsityCsr64)
{
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int64>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->convert_to(sparsity_csr_mtx.get());
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
}


TYPED_TEST(Dense, MovesToSparsityCsr64)
{
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int64>;
    auto sparsity_csr_mtx = SparsityCsr::create(this->mtx4->get_executor());

    this->mtx4->move_to(sparsity_csr_mtx.get());
    auto v = sparsity_csr_mtx->get_const_value();
    auto c = sparsity_csr_mtx->get_const_col_idxs();
    auto r = sparsity_csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(sparsity_csr_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sparsity_csr_mtx->get_num_nonzeros(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
}


TYPED_TEST(Dense, ConvertsToEll32)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->convert_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
}


TYPED_TEST(Dense, MovesToEll32)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->move_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
}


TYPED_TEST(Dense, ConvertsToEll64)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int64>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->convert_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
}


TYPED_TEST(Dense, MovesToEll64)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int64>;
    auto ell_mtx = Ell::create(this->mtx6->get_executor());

    this->mtx6->move_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
}


TYPED_TEST(Dense, ConvertsToEllWithStride)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto ell_mtx =
        Ell::create(this->mtx6->get_executor(), gko::dim<2>{2, 3}, 2, 3);

    this->mtx6->convert_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(ell_mtx->get_stride(), 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], this->invalid_index);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], this->invalid_index);
    EXPECT_EQ(c[5], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{0.0});
    EXPECT_EQ(v[3], T{2.0});
    EXPECT_EQ(v[4], T{0.0});
    EXPECT_EQ(v[5], T{0.0});
}


TYPED_TEST(Dense, MovesToEllWithStride)
{
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto ell_mtx =
        Ell::create(this->mtx6->get_executor(), gko::dim<2>{2, 3}, 2, 3);

    this->mtx6->move_to(ell_mtx.get());
    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(ell_mtx->get_stride(), 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], this->invalid_index);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], this->invalid_index);
    EXPECT_EQ(c[5], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{0.0});
    EXPECT_EQ(v[3], T{2.0});
    EXPECT_EQ(v[4], T{0.0});
    EXPECT_EQ(v[5], T{0.0});
}


TYPED_TEST(Dense, MovesToHybridAutomatically32)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToHybridAutomatically32)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToHybridAutomatically64)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int64>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToHybridAutomatically64)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int64>;
    auto hybrid_mtx = Hybrid::create(this->mtx4->get_executor());

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToHybridWithStrideAutomatically)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 0, 3);

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, ConvertsToHybridWithStrideAutomatically)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 0, 3);

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_coo_values();
    auto c = hybrid_mtx->get_const_coo_col_idxs();
    auto r = hybrid_mtx->get_const_coo_row_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 0);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 4);
    EXPECT_EQ(n, 0);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{3.0});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{5.0});
}


TYPED_TEST(Dense, MovesToHybridWithStrideAndCooLengthByColumns2)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 2, 3, 3,
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 1);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], this->invalid_index);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], this->invalid_index);
    EXPECT_EQ(c[5], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{5.0});
    EXPECT_EQ(v[2], T{0.0});
    EXPECT_EQ(v[3], T{3.0});
    EXPECT_EQ(v[4], T{0.0});
    EXPECT_EQ(v[5], T{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[0], T{2.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[0], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[0], 2);
}


TYPED_TEST(Dense, ConvertsToHybridWithStrideAndCooLengthByColumns2)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 2, 3, 3,
                       std::make_shared<typename Hybrid::column_limit>(2));

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 1);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], this->invalid_index);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], this->invalid_index);
    EXPECT_EQ(c[5], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{5.0});
    EXPECT_EQ(v[2], T{0.0});
    EXPECT_EQ(v[3], T{3.0});
    EXPECT_EQ(v[4], T{0.0});
    EXPECT_EQ(v[5], T{0.0});
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[0], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[0], 2);
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[0], T{2.0});
}


TYPED_TEST(Dense, MovesToHybridWithStrideByPercent40)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 1, 3,
                       std::make_shared<typename Hybrid::imbalance_limit>(0.4));

    this->mtx4->move_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    auto coo_v = hybrid_mtx->get_const_coo_values();
    auto coo_c = hybrid_mtx->get_const_coo_col_idxs();
    auto coo_r = hybrid_mtx->get_const_coo_row_idxs();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 3);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{5.0});
    EXPECT_EQ(v[2], T{0.0});
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(coo_v[0], T{3.0});
    EXPECT_EQ(coo_v[1], T{2.0});
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TYPED_TEST(Dense, ConvertsToHybridWithStrideByPercent40)
{
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto hybrid_mtx =
        Hybrid::create(this->mtx4->get_executor(), gko::dim<2>{2, 3}, 1, 3,
                       std::make_shared<typename Hybrid::imbalance_limit>(0.4));

    this->mtx4->convert_to(hybrid_mtx.get());
    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    auto coo_v = hybrid_mtx->get_const_coo_values();
    auto coo_c = hybrid_mtx->get_const_coo_col_idxs();
    auto coo_r = hybrid_mtx->get_const_coo_row_idxs();

    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 3);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{5.0});
    EXPECT_EQ(v[2], T{0.0});
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(coo_v[0], T{3.0});
    EXPECT_EQ(coo_v[1], T{2.0});
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TYPED_TEST(Dense, ConvertsToSellp32)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->convert_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], this->invalid_index);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], T{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], T{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(Dense, MovesToSellp32)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->move_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], this->invalid_index);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], T{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], T{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(Dense, ConvertsToSellp64)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int64>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->convert_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], this->invalid_index);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], T{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], T{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(Dense, MovesToSellp64)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int64>;
    auto sellp_mtx = Sellp::create(this->mtx7->get_executor());

    this->mtx7->move_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], this->invalid_index);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[gko::matrix::default_slice_size], T{2.0});
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], T{3.0});
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TYPED_TEST(Dense, ConvertsToSellpWithSliceSizeAndStrideFactor)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto sellp_mtx =
        Sellp::create(this->mtx7->get_executor(), gko::dim<2>{}, 2, 2, 0);

    this->mtx7->convert_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 4);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(), 8);
    ASSERT_EQ(sellp_mtx->get_slice_size(), 2);
    ASSERT_EQ(sellp_mtx->get_stride_factor(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], this->invalid_index);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], this->invalid_index);
    EXPECT_EQ(c[6], this->invalid_index);
    EXPECT_EQ(c[7], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
    EXPECT_EQ(v[4], T{3.0});
    EXPECT_EQ(v[5], T{0.0});
    EXPECT_EQ(v[6], T{0.0});
    EXPECT_EQ(v[7], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(l[0], 4);
}


TYPED_TEST(Dense, MovesToSellpWithSliceSizeAndStrideFactor)
{
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto sellp_mtx =
        Sellp::create(this->mtx7->get_executor(), gko::dim<2>{}, 2, 2, 0);

    this->mtx7->move_to(sellp_mtx.get());
    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 4);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(), 8);
    ASSERT_EQ(sellp_mtx->get_slice_size(), 2);
    ASSERT_EQ(sellp_mtx->get_stride_factor(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], this->invalid_index);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], this->invalid_index);
    EXPECT_EQ(c[6], this->invalid_index);
    EXPECT_EQ(c[7], this->invalid_index);
    EXPECT_EQ(v[0], T{1.0});
    EXPECT_EQ(v[1], T{1.5});
    EXPECT_EQ(v[2], T{2.0});
    EXPECT_EQ(v[3], T{0.0});
    EXPECT_EQ(v[4], T{3.0});
    EXPECT_EQ(v[5], T{0.0});
    EXPECT_EQ(v[6], T{0.0});
    EXPECT_EQ(v[7], T{0.0});
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(l[0], 4);
}


TYPED_TEST(Dense, ConvertsToAndFromSellpWithMoreThanOneSlice)
{
    using T = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto x = this->template gen_mtx<Mtx>(65, 25);

    auto sellp_mtx = Sellp::create(this->exec);
    auto dense_mtx = Mtx::create(this->exec);
    x->convert_to(sellp_mtx.get());
    sellp_mtx->convert_to(dense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), x.get(), 0.0);
}


TYPED_TEST(Dense, ConvertsEmptyToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto empty = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToPrecision)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using OtherT = typename gko::next_precision<T>;
    using OtherDense = typename gko::matrix::Dense<OtherT>;
    auto empty = OtherDense::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToCoo)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToCoo)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Coo = typename gko::matrix::Coo<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyMatrixToCsr)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyMatrixToCsr)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Csr = typename gko::matrix::Csr<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToSparsityCsr)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = SparsityCsr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToSparsityCsr)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using SparsityCsr = typename gko::matrix::SparsityCsr<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = SparsityCsr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToEll)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToEll)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Ell = typename gko::matrix::Ell<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Ell::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToHybrid)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToHybrid)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Hybrid = typename gko::matrix::Hybrid<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Hybrid::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, ConvertsEmptyToSellp)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, MovesEmptyToSellp)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using Sellp = typename gko::matrix::Sellp<T, gko::int32>;
    auto empty = Dense::create(this->exec);
    auto res = Sellp::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Dense, SquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = gko::as<Mtx>(this->mtx5->transpose());

    GKO_ASSERT_MTX_NEAR(
        trans, l<T>({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, SquareMatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, this->mtx5->get_size());

    this->mtx5->transpose(trans.get());

    GKO_ASSERT_MTX_NEAR(
        trans, l<T>({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})->transpose(trans.get());

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, -2.0}, {-1.0, 2.0}}), 0.0);
    ASSERT_EQ(trans->get_stride(), 4);
}


TYPED_TEST(Dense, SquareMatrixIsTransposableIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;

    ASSERT_THROW(this->mtx5->transpose(Mtx::create(this->exec).get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = gko::as<Mtx>(this->mtx4->transpose());

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans =
        Mtx::create(this->exec, gko::transpose(this->mtx4->get_size()));

    this->mtx4->transpose(trans.get());

    GKO_ASSERT_MTX_NEAR(trans, l<T>({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}), 0.0);
}


TYPED_TEST(Dense, NonSquareSubmatrixIsTransposableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto trans = Mtx::create(this->exec, gko::dim<2>{2, 1}, 5);

    this->mtx4->create_submatrix({0, 1}, {0, 2})->transpose(trans.get());

    GKO_ASSERT_MTX_NEAR(trans, l({1.0, 3.0}), 0.0);
    ASSERT_EQ(trans->get_stride(), 5);
}


TYPED_TEST(Dense, NonSquareMatrixIsTransposableIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;

    ASSERT_THROW(this->mtx4->transpose(Mtx::create(this->exec).get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixCanGatherRows)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};

    auto row_collection = this->mtx5->row_gather(&permute_idxs);

    GKO_ASSERT_MTX_NEAR(row_collection,
                        l<T>({{-2.0, 2.0, 4.5}, {1.0, -1.0, -0.5}}), 0.0);
}


TYPED_TEST(Dense, SquareMatrixCanGatherRowsIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};
    auto row_collection = Mtx::create(exec, gko::dim<2>{2, 3});

    this->mtx5->row_gather(&permute_idxs, row_collection.get());

    GKO_ASSERT_MTX_NEAR(row_collection,
                        l<T>({{-2.0, 2.0, 4.5}, {1.0, -1.0, -0.5}}), 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixCanGatherRowsIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};
    auto row_collection = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {1, 3})
        ->row_gather(&permute_idxs, row_collection.get());

    GKO_ASSERT_MTX_NEAR(row_collection, l<T>({{2.0, 4.5}, {-1.0, -0.5}}), 0.0);
    ASSERT_EQ(row_collection->get_stride(), 4);
}


TYPED_TEST(Dense, NonSquareSubmatrixCanGatherRowsIntoMixedDense)
{
    using Mtx = typename TestFixture::Mtx;
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int32> gather_index{exec, {1, 0, 1}};
    auto row_collection = MixedMtx::create(exec, gko::dim<2>{3, 3}, 4);

    this->mtx4->row_gather(&gather_index, row_collection.get());

    GKO_ASSERT_MTX_NEAR(
        row_collection,
        l<typename MixedMtx::value_type>(
            {{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}),
        0.0);
}


TYPED_TEST(Dense, NonSquareSubmatrixCanAdvancedGatherRowsIntoMixedDense)
{
    using Mtx = typename TestFixture::Mtx;
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int32> gather_index{exec, {1, 0, 1}};
    auto row_collection = gko::initialize<MixedMtx>(
        {{1.0, 0.5, -1.0}, {-1.5, 0.5, 1.0}, {2.0, -3.0, 1.0}}, exec);
    auto alpha = gko::initialize<MixedMtx>({1.0}, exec);
    auto beta = gko::initialize<Mtx>({2.0}, exec);

    this->mtx4->row_gather(alpha.get(), &gather_index, beta.get(),
                           row_collection.get());

    GKO_ASSERT_MTX_NEAR(
        row_collection,
        l<typename MixedMtx::value_type>(
            {{2.0, 6.0, -2.0}, {-2.0, 4.0, 4.0}, {4.0, -1.0, 2.0}}),
        0.0);
}


TYPED_TEST(Dense, SquareMatrixGatherRowsIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};

    ASSERT_THROW(this->mtx5->row_gather(&permute_idxs, Mtx::create(exec).get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixCanGatherRows64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};

    auto row_collection = this->mtx5->row_gather(&permute_idxs);

    GKO_ASSERT_MTX_NEAR(row_collection,
                        l<T>({{-2.0, 2.0, 4.5}, {1.0, -1.0, -0.5}}), 0.0);
}


TYPED_TEST(Dense, SquareMatrixCanGatherRowsIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};
    auto row_collection = Mtx::create(exec, gko::dim<2>{2, 3});

    this->mtx5->row_gather(&permute_idxs, row_collection.get());

    GKO_ASSERT_MTX_NEAR(row_collection,
                        l<T>({{-2.0, 2.0, 4.5}, {1.0, -1.0, -0.5}}), 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixCanGatherRowsIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};
    auto row_collection = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {1, 3})
        ->row_gather(&permute_idxs, row_collection.get());

    GKO_ASSERT_MTX_NEAR(row_collection, l<T>({{2.0, 4.5}, {-1.0, -0.5}}), 0.0);
    ASSERT_EQ(row_collection->get_stride(), 4);
}


TYPED_TEST(Dense, NonSquareSubmatrixCanGatherRowsIntoMixedDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int64> gather_index{exec, {1, 0, 1}};
    auto row_collection = MixedMtx::create(exec, gko::dim<2>{3, 3}, 4);

    this->mtx4->row_gather(&gather_index, row_collection.get());

    GKO_ASSERT_MTX_NEAR(
        row_collection,
        l<typename MixedMtx::value_type>(
            {{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}),
        0.0);
}


TYPED_TEST(Dense, SquareMatrixGatherRowsIntoDenseFailsForWrongDimensions64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};

    ASSERT_THROW(this->mtx5->row_gather(&permute_idxs, Mtx::create(exec).get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    auto permuted = gko::as<Mtx>(this->mtx5->permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(Dense, SquareMatrixIsPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    this->mtx5->permute(&permute_idxs, permuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);
    auto mtx = this->mtx5->create_submatrix({0, 2}, {1, 3});

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(mtx->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    mtx->permute(&permute_idxs, permuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(Dense, NonSquareMatrixPermuteIntoDenseFails)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx4->permute(&permute_idxs, this->mtx4->clone().get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2}};

    ASSERT_THROW(this->mtx5->permute(&permute_idxs, this->mtx5->clone().get()),
                 gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->permute(&permute_idxs, Mtx::create(exec).get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsInversePermutable)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto ref_permuted = gko::as<Mtx>(
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&permute_idxs))
            ->inverse_column_permute(&permute_idxs));
    auto permuted = gko::as<Mtx>(this->mtx5->inverse_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(Dense, SquareMatrixIsInversePermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    auto ref_permuted = gko::as<Mtx>(
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&permute_idxs))
            ->inverse_column_permute(&permute_idxs));
    this->mtx5->inverse_permute(&permute_idxs, permuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsInversePermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);
    auto mtx = this->mtx5->create_submatrix({0, 2}, {1, 3});

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(mtx->inverse_row_permute(&permute_idxs))
                         ->inverse_column_permute(&permute_idxs));
    mtx->inverse_permute(&permute_idxs, permuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(Dense, NonSquareMatrixInversePermuteIntoDenseFails)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx4->inverse_permute(&permute_idxs, this->mtx4->clone().get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense,
           SquareMatrixInversePermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {0, 1}};

    ASSERT_THROW(
        this->mtx5->inverse_permute(&permute_idxs, this->mtx5->clone().get()),
        gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixInversePermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->inverse_permute(&permute_idxs, Mtx::create(exec).get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    auto permuted = gko::as<Mtx>(this->mtx5->permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(Dense, SquareMatrixIsPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    this->mtx5->permute(&permute_idxs, permuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);
    auto mtx = this->mtx5->create_submatrix({0, 2}, {1, 3});

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(mtx->row_permute(&permute_idxs))
                         ->column_permute(&permute_idxs));
    mtx->permute(&permute_idxs, permuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(Dense, NonSquareMatrixPermuteIntoDenseFails64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx4->permute(&permute_idxs, this->mtx4->clone().get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixPermuteIntoDenseFailsForWrongPermutationSize64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2}};

    ASSERT_THROW(this->mtx5->permute(&permute_idxs, this->mtx5->clone().get()),
                 gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixPermuteIntoDenseFailsForWrongDimensions64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->permute(&permute_idxs, Mtx::create(exec).get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsInversePermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    auto ref_permuted = gko::as<Mtx>(
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&permute_idxs))
            ->inverse_column_permute(&permute_idxs));
    auto permuted = gko::as<Mtx>(this->mtx5->inverse_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(Dense, SquareMatrixIsInversePermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};
    auto permuted = Mtx::create(exec, this->mtx5->get_size());

    auto ref_permuted = gko::as<Mtx>(
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&permute_idxs))
            ->inverse_column_permute(&permute_idxs));
    this->mtx5->inverse_permute(&permute_idxs, permuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsInversePermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};
    auto permuted = Mtx::create(exec, gko::dim<2>{2, 2}, 4);
    auto mtx = this->mtx5->create_submatrix({0, 2}, {1, 3});

    auto ref_permuted =
        gko::as<Mtx>(gko::as<Mtx>(mtx->inverse_row_permute(&permute_idxs))
                         ->inverse_column_permute(&permute_idxs));
    mtx->inverse_permute(&permute_idxs, permuted.get());

    GKO_ASSERT_MTX_NEAR(permuted, ref_permuted, 0.0);
    ASSERT_EQ(permuted->get_stride(), 4);
}


TYPED_TEST(Dense, NonSquareMatrixInversePermuteIntoDenseFails64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx4->inverse_permute(&permute_idxs, this->mtx4->clone().get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense,
           SquareMatrixInversePermuteIntoDenseFailsForWrongPermutationSize64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2}};

    ASSERT_THROW(
        this->mtx5->inverse_permute(&permute_idxs, this->mtx5->clone().get()),
        gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixInversePermuteIntoDenseFailsForWrongDimensions64)
{
    using Mtx = typename TestFixture::Mtx;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->inverse_permute(&permute_idxs, Mtx::create(exec).get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsRowPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto row_permute = gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        row_permute,
        l<T>({{-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsRowPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};

    auto row_permute = gko::as<Mtx>(this->mtx4->row_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(row_permute, l<T>({{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}}),
                        0.0);
}


TYPED_TEST(Dense, SquareMatrixIsRowPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->row_permute(&permute_idxs, row_permute.get());

    GKO_ASSERT_MTX_NEAR(
        row_permute,
        l<T>({{-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}}), 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsRowPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};
    auto row_permute = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->row_permute(&permute_idxs, row_permute.get());

    GKO_ASSERT_MTX_NEAR(row_permute, l<T>({{-2.0, 2.0}, {1.0, -1.0}}), 0.0);
    ASSERT_EQ(row_permute->get_stride(), 4);
}


TYPED_TEST(Dense, SquareMatrixRowPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->row_permute(&permute_idxs, row_permute.get()),
                 gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixRowPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->row_permute(&permute_idxs, Mtx::create(exec).get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsColPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = gko::as<Mtx>(this->mtx5->column_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        c_permute, l<T>({{-1.0, -0.5, 1.0}, {2.0, 4.5, -2.0}, {3.4, 1.2, 2.1}}),
        0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsColPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = gko::as<Mtx>(this->mtx4->column_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(c_permute, l<T>({{3.0, 2.0, 1.0}, {5.0, 0.0, 0.0}}),
                        0.0);
}


TYPED_TEST(Dense, SquareMatrixIsColPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};
    auto c_permute = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->column_permute(&permute_idxs, c_permute.get());

    GKO_ASSERT_MTX_NEAR(
        c_permute, l<T>({{-1.0, -0.5, 1.0}, {2.0, 4.5, -2.0}, {3.4, 1.2, 2.1}}),
        0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsColPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};
    auto c_permute = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->column_permute(&permute_idxs, c_permute.get());

    GKO_ASSERT_MTX_NEAR(c_permute, l<T>({{-1.0, 1.0}, {2.0, -2.0}}), 0.0);
    ASSERT_EQ(c_permute->get_stride(), 4);
}


TYPED_TEST(Dense, SquareMatrixColPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->column_permute(&permute_idxs, row_permute.get()),
                 gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixColPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->column_permute(&permute_idxs, Mtx::create(exec).get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsInverseRowPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_row_permute =
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        inverse_row_permute,
        l<T>({{2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsInverseRowPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int32> inverse_permute_idxs{exec, {1, 0}};

    auto inverse_row_permute =
        gko::as<Mtx>(this->mtx4->inverse_row_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(inverse_row_permute,
                        l<T>({{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}}), 0.0);
}


TYPED_TEST(Dense, SquareMatrixIsInverseRowPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->inverse_row_permute(&permute_idxs, row_permute.get());

    GKO_ASSERT_MTX_NEAR(
        row_permute,
        l<T>({{2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}}), 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsInverseRowPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};
    auto row_permute = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->inverse_row_permute(&permute_idxs, row_permute.get());

    GKO_ASSERT_MTX_NEAR(row_permute, l<T>({{-2.0, 2.0}, {1.0, -1.0}}), 0.0);
    ASSERT_EQ(row_permute->get_stride(), 4);
}


TYPED_TEST(Dense,
           SquareMatrixInverseRowPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(
        this->mtx5->inverse_row_permute(&permute_idxs, row_permute.get()),
        gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixInverseRowPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->inverse_row_permute(&permute_idxs, Mtx::create(exec).get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsInverseColPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        gko::as<Mtx>(this->mtx5->inverse_column_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        inverse_c_permute,
        l<T>({{-0.5, 1.0, -1.0}, {4.5, -2.0, 2.0}, {1.2, 2.1, 3.4}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsInverseColPermutable)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        gko::as<Mtx>(this->mtx4->inverse_column_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(inverse_c_permute,
                        l<T>({{2.0, 1.0, 3.0}, {0.0, 0.0, 5.0}}), 0.0);
}


TYPED_TEST(Dense, SquareMatrixIsInverseColPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};
    auto c_permute = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->inverse_column_permute(&permute_idxs, c_permute.get());

    GKO_ASSERT_MTX_NEAR(
        c_permute, l<T>({{-0.5, 1.0, -1.0}, {4.5, -2.0, 2.0}, {1.2, 2.1, 3.4}}),
        0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsInverseColPermutableIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 0}};
    auto c_permute = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->column_permute(&permute_idxs, c_permute.get());

    GKO_ASSERT_MTX_NEAR(c_permute, l<T>({{-1.0, 1.0}, {2.0, -2.0}}), 0.0);
    ASSERT_EQ(c_permute->get_stride(), 4);
}


TYPED_TEST(Dense,
           SquareMatrixInverseColPermuteIntoDenseFailsForWrongPermutationSize)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(
        this->mtx5->inverse_column_permute(&permute_idxs, row_permute.get()),
        gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixInverseColPermuteIntoDenseFailsForWrongDimensions)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->inverse_column_permute(&permute_idxs,
                                                    Mtx::create(exec).get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsRowPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    auto row_permute = gko::as<Mtx>(this->mtx5->row_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        row_permute,
        l<T>({{-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsRowPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};

    auto row_permute = gko::as<Mtx>(this->mtx4->row_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(row_permute, l<T>({{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}}),
                        0.0);
}


TYPED_TEST(Dense, SquareMatrixIsRowPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->row_permute(&permute_idxs, row_permute.get());

    GKO_ASSERT_MTX_NEAR(
        row_permute,
        l<T>({{-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}}), 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsRowPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};
    auto row_permute = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->row_permute(&permute_idxs, row_permute.get());

    GKO_ASSERT_MTX_NEAR(row_permute, l<T>({{-2.0, 2.0}, {1.0, -1.0}}), 0.0);
    ASSERT_EQ(row_permute->get_stride(), 4);
}


TYPED_TEST(Dense, SquareMatrixRowPermuteIntoDenseFailsForWrongPermutationSize64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->row_permute(&permute_idxs, row_permute.get()),
                 gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixRowPermuteIntoDenseFailsForWrongDimensions64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->row_permute(&permute_idxs, Mtx::create(exec).get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsColPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = gko::as<Mtx>(this->mtx5->column_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        c_permute, l<T>({{-1.0, -0.5, 1.0}, {2.0, 4.5, -2.0}, {3.4, 1.2, 2.1}}),
        0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsColPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = gko::as<Mtx>(this->mtx4->column_permute(&permute_idxs));

    GKO_ASSERT_MTX_NEAR(c_permute, l<T>({{3.0, 2.0, 1.0}, {5.0, 0.0, 0.0}}),
                        0.0);
}


TYPED_TEST(Dense, SquareMatrixIsColPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};
    auto c_permute = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->column_permute(&permute_idxs, c_permute.get());

    GKO_ASSERT_MTX_NEAR(
        c_permute, l<T>({{-1.0, -0.5, 1.0}, {2.0, 4.5, -2.0}, {3.4, 1.2, 2.1}}),
        0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsColPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};
    auto c_permute = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->column_permute(&permute_idxs, c_permute.get());

    GKO_ASSERT_MTX_NEAR(c_permute, l<T>({{-1.0, 1.0}, {2.0, -2.0}}), 0.0);
    ASSERT_EQ(c_permute->get_stride(), 4);
}


TYPED_TEST(Dense, SquareMatrixColPermuteIntoDenseFailsForWrongPermutationSize64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(this->mtx5->column_permute(&permute_idxs, row_permute.get()),
                 gko::ValueMismatch);
}


TYPED_TEST(Dense, SquareMatrixColPermuteIntoDenseFailsForWrongDimensions64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->column_permute(&permute_idxs, Mtx::create(exec).get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsInverseRowPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_row_permute =
        gko::as<Mtx>(this->mtx5->inverse_row_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        inverse_row_permute,
        l<T>({{2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsInverseRowPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int64> inverse_permute_idxs{exec, {1, 0}};

    auto inverse_row_permute =
        gko::as<Mtx>(this->mtx4->inverse_row_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(inverse_row_permute,
                        l<T>({{0.0, 5.0, 0.0}, {1.0, 3.0, 2.0}}), 0.0);
}


TYPED_TEST(Dense, SquareMatrixIsInverseRowPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->inverse_row_permute(&permute_idxs, row_permute.get());

    GKO_ASSERT_MTX_NEAR(
        row_permute,
        l<T>({{2.1, 3.4, 1.2}, {1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}}), 0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsInverseRowPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};
    auto row_permute = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->inverse_row_permute(&permute_idxs, row_permute.get());

    GKO_ASSERT_MTX_NEAR(row_permute, l<T>({{-2.0, 2.0}, {1.0, -1.0}}), 0.0);
    ASSERT_EQ(row_permute->get_stride(), 4);
}


TYPED_TEST(Dense,
           SquareMatrixInverseRowPermuteIntoDenseFailsForWrongPermutationSize64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(
        this->mtx5->inverse_row_permute(&permute_idxs, row_permute.get()),
        gko::ValueMismatch);
}


TYPED_TEST(Dense,
           SquareMatrixInverseRowPermuteIntoDenseFailsForWrongDimensions64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(
        this->mtx5->inverse_row_permute(&permute_idxs, Mtx::create(exec).get()),
        gko::DimensionMismatch);
}


TYPED_TEST(Dense, SquareMatrixIsInverseColPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        gko::as<Mtx>(this->mtx5->inverse_column_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(
        inverse_c_permute,
        l<T>({{-0.5, 1.0, -1.0}, {4.5, -2.0, 2.0}, {1.2, 2.1, 3.4}}), 0.0);
}


TYPED_TEST(Dense, NonSquareMatrixIsInverseColPermutable64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx4->get_executor();
    gko::array<gko::int64> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        gko::as<Mtx>(this->mtx4->inverse_column_permute(&inverse_permute_idxs));

    GKO_ASSERT_MTX_NEAR(inverse_c_permute,
                        l<T>({{2.0, 1.0, 3.0}, {0.0, 0.0, 5.0}}), 0.0);
}


TYPED_TEST(Dense, SquareMatrixIsInverseColPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};
    auto c_permute = Mtx::create(exec, this->mtx5->get_size());

    this->mtx5->inverse_column_permute(&permute_idxs, c_permute.get());

    GKO_ASSERT_MTX_NEAR(
        c_permute, l<T>({{-0.5, 1.0, -1.0}, {4.5, -2.0, 2.0}, {1.2, 2.1, 3.4}}),
        0.0);
}


TYPED_TEST(Dense, SquareSubmatrixIsInverseColPermutableIntoDense64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 0}};
    auto c_permute = Mtx::create(exec, gko::dim<2>{2, 2}, 4);

    this->mtx5->create_submatrix({0, 2}, {0, 2})
        ->column_permute(&permute_idxs, c_permute.get());

    GKO_ASSERT_MTX_NEAR(c_permute, l<T>({{-1.0, 1.0}, {2.0, -2.0}}), 0.0);
    ASSERT_EQ(c_permute->get_stride(), 4);
}


TYPED_TEST(Dense,
           SquareMatrixInverseColPermuteIntoDenseFailsForWrongPermutationSize64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2}};
    auto row_permute = Mtx::create(exec, this->mtx5->get_size());

    ASSERT_THROW(
        this->mtx5->inverse_column_permute(&permute_idxs, row_permute.get()),
        gko::ValueMismatch);
}


TYPED_TEST(Dense,
           SquareMatrixInverseColPermuteIntoDenseFailsForWrongDimensions64)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = this->mtx5->get_executor();
    gko::array<gko::int64> permute_idxs{exec, {1, 2, 0}};

    ASSERT_THROW(this->mtx5->inverse_column_permute(&permute_idxs,
                                                    Mtx::create(exec).get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, ExtractsDiagonalFromSquareMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx5->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 3);
    ASSERT_EQ(diag->get_size()[1], 3);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
    ASSERT_EQ(diag->get_values()[2], T{1.2});
}


TYPED_TEST(Dense, ExtractsDiagonalFromTallSkinnyMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx4->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromShortFatMatrix)
{
    using T = typename TestFixture::value_type;

    auto diag = this->mtx8->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromSquareMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 3);

    this->mtx5->extract_diagonal(diag.get());

    ASSERT_EQ(diag->get_size()[0], 3);
    ASSERT_EQ(diag->get_size()[1], 3);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
    ASSERT_EQ(diag->get_values()[2], T{1.2});
}


TYPED_TEST(Dense, ExtractsDiagonalFromTallSkinnyMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2);

    this->mtx4->extract_diagonal(diag.get());

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Dense, ExtractsDiagonalFromShortFatMatrixIntoDiagonal)
{
    using T = typename TestFixture::value_type;
    auto diag = gko::matrix::Diagonal<T>::create(this->exec, 2);

    this->mtx8->extract_diagonal(diag.get());

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{2.});
}


TYPED_TEST(Dense, InplaceAbsolute)
{
    using T = typename TestFixture::value_type;

    this->mtx5->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        this->mtx5, l<T>({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, InplaceAbsoluteSubMatrix)
{
    using T = typename TestFixture::value_type;
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        this->mtx5, l<T>({{1.0, 1.0, -0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, OutplaceAbsolute)
{
    using T = typename TestFixture::value_type;

    auto abs_mtx = this->mtx5->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, OutplaceAbsoluteIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto abs_mtx =
        gko::remove_complex<Mtx>::create(this->exec, this->mtx5->get_size());

    this->mtx5->compute_absolute(abs_mtx.get());

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 1.0, 0.5}, {2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}),
        0.0);
}


TYPED_TEST(Dense, OutplaceAbsoluteSubMatrix)
{
    using T = typename TestFixture::value_type;
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, l<T>({{1.0, 1.0}, {2.0, 2.0}}), 0);
    GKO_ASSERT_EQ(abs_mtx->get_stride(), 2);
}


TYPED_TEST(Dense, OutplaceSubmatrixAbsoluteIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto mtx = this->mtx5->create_submatrix(gko::span{0, 2}, gko::span{0, 2});
    auto abs_mtx =
        gko::remove_complex<Mtx>::create(this->exec, gko::dim<2>{2, 2}, 4);

    mtx->compute_absolute(abs_mtx.get());

    GKO_ASSERT_MTX_NEAR(abs_mtx, l<T>({{1.0, 1.0}, {2.0, 2.0}}), 0);
    GKO_ASSERT_EQ(abs_mtx->get_stride(), 4);
}


TYPED_TEST(Dense, AppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();
    auto b =
        gko::initialize<Vec>({{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
                              {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
                              {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}},
                             exec);
    auto x = Vec::create(exec, gko::dim<2>{2, 2});

    this->mtx1->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{14.0, 16.0}, complex_type{20.0, 22.0}},
           {complex_type{17.0, 19.0}, complex_type{24.5, 26.5}}}),
        0.0);
}


TYPED_TEST(Dense, AppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using Vec = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();
    auto b = gko::initialize<Vec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}},
        exec);
    auto x = Vec::create(exec, gko::dim<2>{2, 2});

    this->mtx1->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{14.0, 16.0}, mixed_complex_type{20.0, 22.0}},
           {mixed_complex_type{17.0, 19.0}, mixed_complex_type{24.5, 26.5}}}),
        0.0);
}


TYPED_TEST(Dense, AdvancedAppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using DenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    auto b = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}},
        exec);
    auto x = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}},
        exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    auto beta = gko::initialize<Dense>({2.0}, this->exec);

    this->mtx1->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-12.0, -16.0}, complex_type{-16.0, -20.0}},
           {complex_type{-13.0, -15.0}, complex_type{-18.5, -20.5}}}),
        0.0);
}


TYPED_TEST(Dense, AdvancedAppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedDense = gko::matrix::Dense<mixed_value_type>;
    using MixedDenseComplex = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    auto b = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}},
        exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}},
        exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);

    this->mtx1->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{-12.0, -16.0}, mixed_complex_type{-16.0, -20.0}},
           {mixed_complex_type{-13.0, -15.0},
            mixed_complex_type{-18.5, -20.5}}}),
        0.0);
}


TYPED_TEST(Dense, MakeComplex)
{
    using T = typename TestFixture::value_type;

    auto complex_mtx = this->mtx5->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_mtx, this->mtx5, 0.0);
}


TYPED_TEST(Dense, MakeComplexIntoDense)
{
    using T = typename TestFixture::value_type;
    using ComplexMtx = typename TestFixture::ComplexMtx;
    auto exec = this->mtx5->get_executor();

    auto complex_mtx = ComplexMtx::create(exec, this->mtx5->get_size());
    this->mtx5->make_complex(complex_mtx.get());

    GKO_ASSERT_MTX_NEAR(complex_mtx, this->mtx5, 0.0);
}


TYPED_TEST(Dense, MakeComplexIntoDenseFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using ComplexMtx = typename TestFixture::ComplexMtx;
    auto exec = this->mtx5->get_executor();

    auto complex_mtx = ComplexMtx::create(exec);

    ASSERT_THROW(this->mtx5->make_complex(complex_mtx.get()),
                 gko::DimensionMismatch);
}


TYPED_TEST(Dense, GetReal)
{
    using T = typename TestFixture::value_type;

    auto real_mtx = this->mtx5->get_real();

    GKO_ASSERT_MTX_NEAR(real_mtx, this->mtx5, 0.0);
}


TYPED_TEST(Dense, GetRealIntoDense)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto real_mtx = RealMtx::create(exec, this->mtx5->get_size());
    this->mtx5->get_real(real_mtx.get());

    GKO_ASSERT_MTX_NEAR(real_mtx, this->mtx5, 0.0);
}


TYPED_TEST(Dense, GetRealIntoDenseFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto real_mtx = RealMtx::create(exec);
    ASSERT_THROW(this->mtx5->get_real(real_mtx.get()), gko::DimensionMismatch);
}


TYPED_TEST(Dense, GetImag)
{
    using T = typename TestFixture::value_type;

    auto imag_mtx = this->mtx5->get_imag();

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}),
        0.0);
}


TYPED_TEST(Dense, GetImagIntoDense)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto imag_mtx = RealMtx::create(exec, this->mtx5->get_size());
    this->mtx5->get_imag(imag_mtx.get());

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}),
        0.0);
}


TYPED_TEST(Dense, GetImagIntoDenseFailsForWrongDimensions)
{
    using T = typename TestFixture::value_type;
    using RealMtx = typename TestFixture::RealMtx;
    auto exec = this->mtx5->get_executor();

    auto imag_mtx = RealMtx::create(exec);
    ASSERT_THROW(this->mtx5->get_imag(imag_mtx.get()), gko::DimensionMismatch);
}


TYPED_TEST(Dense, MakeTemporaryConversionDoesntConvertOnMatch)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Mtx>({8.0}, this->exec);

    ASSERT_EQ(gko::make_temporary_conversion<T>(alpha.get()).get(),
              alpha.get());
}


TYPED_TEST(Dense, MakeTemporaryConversionConvertsBack)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    using MixedT = typename MixedMtx::value_type;
    auto alpha = gko::initialize<MixedMtx>({8.0}, this->exec);

    {
        auto conversion = gko::make_temporary_conversion<T>(alpha.get());
        conversion->at(0, 0) = T{7.0};
    }

    ASSERT_EQ(alpha->at(0, 0), MixedT{7.0});
}


TYPED_TEST(Dense, MakeTemporaryConversionConstDoesntConvertBack)
{
    using MixedMtx = typename TestFixture::MixedMtx;
    using T = typename TestFixture::value_type;
    using MixedT = typename MixedMtx::value_type;
    auto alpha = gko::initialize<MixedMtx>({8.0}, this->exec);

    {
        auto conversion = gko::make_temporary_conversion<T>(
            static_cast<const MixedMtx*>(alpha.get()));
        alpha->at(0, 0) = MixedT{7.0};
    }

    ASSERT_EQ(alpha->at(0, 0), MixedT{7.0});
}


TYPED_TEST(Dense, ScaleAddIdentityRectangular)
{
    using T = typename TestFixture::value_type;
    using Vec = typename TestFixture::Mtx;
    using MixedVec = typename TestFixture::MixedMtx;
    auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto b = gko::initialize<Vec>(
        {I<T>{2.0, 0.0}, I<T>{1.0, 2.5}, I<T>{0.0, -4.0}}, this->exec);

    b->add_scaled_identity(alpha.get(), beta.get());

    GKO_ASSERT_MTX_NEAR(b, l({{0.0, 0.0}, {-1.0, -0.5}, {0.0, 4.0}}), 0.0);
}


template <typename T>
class DenseComplex : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using RealMtx = gko::matrix::Dense<gko::remove_complex<value_type>>;
};


TYPED_TEST_SUITE(DenseComplex, gko::test::ComplexValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(DenseComplex, ScalesWithRealScalar)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto alpha =
        gko::initialize<RealDense>({gko::remove_complex<T>{-2.0}}, exec);

    mtx->scale(alpha.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{2.0, -4.5}},
                              {T{4.0, -3.0}, T{-9.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, -2.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, ScalesWithRealVector)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto alpha = gko::initialize<RealDense>({{RealT{-2.0}, RealT{4.0}}}, exec);

    mtx->scale(alpha.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{-4.0, 9.0}},
                              {T{4.0, -3.0}, T{18.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, 4.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, InvScalesWithRealScalar)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto alpha =
        gko::initialize<RealDense>({gko::remove_complex<T>{-0.5}}, exec);

    mtx->inv_scale(alpha.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{2.0, -4.5}},
                              {T{4.0, -3.0}, T{-9.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, -2.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, InvScalesWithRealVector)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto alpha = gko::initialize<RealDense>({{RealT{-0.5}, RealT{0.25}}}, exec);

    mtx->inv_scale(alpha.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-2.0, -4.0}, T{-4.0, 9.0}},
                              {T{4.0, -3.0}, T{18.0, 0.0}},
                              {T{-2.0, 0.0}, T{0.0, 4.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, AddsScaledWithRealScalar)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto mtx2 = gko::initialize<Dense>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                        {T{3.0, 1.0}, T{0.0, 2.0}},
                                        {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                       exec);
    auto alpha =
        gko::initialize<RealDense>({gko::remove_complex<T>{-2.0}}, exec);

    mtx->add_scaled(alpha.get(), mtx2.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{-11.0, -0.75}},
                              {T{-8.0, -0.5}, T{4.5, -4.0}},
                              {T{3.0, -2.0}, T{-1.0, 5.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, AddsScaledWithRealVector)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto mtx2 = gko::initialize<Dense>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                        {T{3.0, 1.0}, T{0.0, 2.0}},
                                        {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                       exec);
    auto alpha = gko::initialize<RealDense>({{RealT{-2.0}, RealT{4.0}}}, exec);

    mtx->add_scaled(alpha.get(), mtx2.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{19.0, 8.25}},
                              {T{-8.0, -0.5}, T{4.5, 8.0}},
                              {T{3.0, -2.0}, T{2.0, -7.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, SubtractsScaledWithRealScalar)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto mtx2 = gko::initialize<Dense>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                        {T{3.0, 1.0}, T{0.0, 2.0}},
                                        {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                       exec);
    auto alpha =
        gko::initialize<RealDense>({gko::remove_complex<T>{2.0}}, exec);

    mtx->sub_scaled(alpha.get(), mtx2.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{-11.0, -0.75}},
                              {T{-8.0, -0.5}, T{4.5, -4.0}},
                              {T{3.0, -2.0}, T{-1.0, 5.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, SubtractsScaledWithRealVector)
{
    using Dense = typename TestFixture::Mtx;
    using RealDense = gko::remove_complex<Dense>;
    using T = typename TestFixture::value_type;
    using RealT = gko::remove_complex<T>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.25}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto mtx2 = gko::initialize<Dense>({{T{4.0, -1.0}, T{5.0, 1.5}},
                                        {T{3.0, 1.0}, T{0.0, 2.0}},
                                        {T{-1.0, 1.0}, T{0.5, -2.0}}},
                                       exec);
    auto alpha = gko::initialize<RealDense>({{RealT{2.0}, RealT{-4.0}}}, exec);

    mtx->sub_scaled(alpha.get(), mtx2.get());

    GKO_ASSERT_MTX_NEAR(mtx,
                        l<T>({{T{-7.0, 4.0}, T{19.0, 8.25}},
                              {T{-8.0, -0.5}, T{4.5, 8.0}},
                              {T{3.0, -2.0}, T{2.0, -7.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, NonSquareMatrixIsConjugateTransposable)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.1}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);

    auto trans = gko::as<Dense>(mtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(trans,
                        l<T>({{T{1.0, -2.0}, T{-2.0, -1.5}, T{1.0, 0.0}},
                              {T{-1.0, -2.1}, T{4.5, 0.0}, T{0.0, -1.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, NonSquareMatrixIsConjugateTransposableIntoDense)
{
    using Dense = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Dense>({{T{1.0, 2.0}, T{-1.0, 2.1}},
                                       {T{-2.0, 1.5}, T{4.5, 0.0}},
                                       {T{1.0, 0.0}, T{0.0, 1.0}}},
                                      exec);
    auto trans = Dense::create(exec, gko::transpose(mtx->get_size()));

    mtx->conj_transpose(trans.get());

    GKO_ASSERT_MTX_NEAR(trans,
                        l<T>({{T{1.0, -2.0}, T{-2.0, -1.5}, T{1.0, 0.0}},
                              {T{-1.0, -2.1}, T{4.5, 0.0}, T{0.0, -1.0}}}),
                        0.0);
}


TYPED_TEST(DenseComplex, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l<T>({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


TYPED_TEST(DenseComplex, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, OutplaceAbsoluteIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);
    auto abs_mtx = gko::remove_complex<Mtx>::create(exec, mtx->get_size());

    mtx->compute_absolute(abs_mtx.get());

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l<T>({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, MakeComplex)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto complex_mtx = mtx->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_mtx, mtx, 0.0);
}


TYPED_TEST(DenseComplex, MakeComplexIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto complex_mtx = Mtx::create(exec, mtx->get_size());
    mtx->make_complex(complex_mtx.get());

    GKO_ASSERT_MTX_NEAR(complex_mtx, mtx, 0.0);
}


TYPED_TEST(DenseComplex, GetReal)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto real_mtx = mtx->get_real();

    GKO_ASSERT_MTX_NEAR(
        real_mtx, l<T>({{1.0, 3.0, 0.0}, {-4.0, -1.0, 0.0}, {0.0, 0.0, 2.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, GetRealIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using RealMtx = typename TestFixture::RealMtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto real_mtx = RealMtx::create(exec, mtx->get_size());
    mtx->get_real(real_mtx.get());

    GKO_ASSERT_MTX_NEAR(
        real_mtx, l<T>({{1.0, 3.0, 0.0}, {-4.0, -1.0, 0.0}, {0.0, 0.0, 2.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, GetImag)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto imag_mtx = mtx->get_imag();

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 4.0, 2.0}, {-3.0, 0.0, 0.0}, {0.0, -1.5, 0.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, GetImagIntoDense)
{
    using Mtx = typename TestFixture::Mtx;
    using RealMtx = typename TestFixture::RealMtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
                                     {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
                                     {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}},
                                    exec);

    auto imag_mtx = RealMtx::create(exec, mtx->get_size());
    mtx->get_imag(imag_mtx.get());

    GKO_ASSERT_MTX_NEAR(
        imag_mtx, l<T>({{0.0, 4.0, 2.0}, {-3.0, 0.0, 0.0}, {0.0, -1.5, 0.0}}),
        0.0);
}


TYPED_TEST(DenseComplex, Dot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto a =
        gko::initialize<Mtx>({T{1.0, 0.0}, T{3.0, 4.0}, T{1.0, 2.0}}, exec);
    auto b =
        gko::initialize<Mtx>({T{1.0, -2.0}, T{5.0, 0.0}, T{0.0, -3.0}}, exec);
    auto result = gko::initialize<Mtx>({T{0.0, 0.0}}, exec);

    a->compute_dot(b.get(), result.get());

    GKO_ASSERT_MTX_NEAR(result, l({T{22.0, 15.0}}), 0.0);
}


TYPED_TEST(DenseComplex, ConjDot)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    auto exec = gko::ReferenceExecutor::create();
    auto a =
        gko::initialize<Mtx>({T{1.0, 0.0}, T{3.0, 4.0}, T{1.0, 2.0}}, exec);
    auto b =
        gko::initialize<Mtx>({T{1.0, -2.0}, T{5.0, 0.0}, T{0.0, -3.0}}, exec);
    auto result = gko::initialize<Mtx>({T{0.0, 0.0}}, exec);

    a->compute_conj_dot(b.get(), result.get());

    GKO_ASSERT_MTX_NEAR(result, l({T{10.0, -25.0}}), 0.0);
}


}  // namespace
