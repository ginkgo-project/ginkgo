/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/matrix/dense_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/executor.hpp"


namespace {


class Dense : public ::testing::Test {
protected:
    using itype = int;
#if GINKGO_COMMON_SINGLE_MODE
    using vtype = float;
#else
    using vtype = double;
#endif
    // in single mode, mixed_type will be the same as vtype
    using mixed_type = float;
    using Mtx = gko::matrix::Dense<vtype>;
    using MixedMtx = gko::matrix::Dense<mixed_type>;
    using NormVector = gko::matrix::Dense<gko::remove_complex<vtype>>;
    using Arr = gko::Array<itype>;
    using ComplexMtx = gko::matrix::Dense<std::complex<vtype>>;
    using Diagonal = gko::matrix::Diagonal<vtype>;
    using MixedComplexMtx = gko::matrix::Dense<std::complex<mixed_type>>;

    Dense() : rand_engine(15) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        init_executor(ref, exec);
    }

    void TearDown()
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_vector_data(gko::size_type num_vecs,
                            bool different_alpha = false)
    {
        x = gen_mtx<Mtx>(1000, num_vecs);
        y = gen_mtx<Mtx>(1000, num_vecs);
        c_x = gen_mtx<ComplexMtx>(1000, num_vecs);
        c_y = gen_mtx<ComplexMtx>(1000, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(1, num_vecs);
            c_alpha = gen_mtx<ComplexMtx>(1, num_vecs);
        } else {
            alpha = gko::initialize<Mtx>({2.0}, ref);
            c_alpha =
                gko::initialize<ComplexMtx>({std::complex<vtype>{2.0}}, ref);
        }
        dx = gko::clone(exec, x);
        dy = gko::clone(exec, y);
        dc_x = gko::clone(exec, c_x);
        dc_y = gko::clone(exec, c_y);
        dalpha = gko::clone(exec, alpha);
        dc_alpha = gko::clone(exec, c_alpha);
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(40, 25);
        y = gen_mtx<Mtx>(25, 35);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        square = gen_mtx<Mtx>(x->get_size()[0], x->get_size()[0]);
        dx = gko::clone(exec, x);
        dy = gko::clone(exec, y);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
        dsquare = gko::clone(exec, square);

        std::vector<int> tmp(x->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<int> tmp2(x->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        std::vector<int> tmp3(x->get_size()[0] / 10);
        std::uniform_int_distribution<int> row_dist(0, x->get_size()[0] - 1);
        for (auto& i : tmp3) {
            i = row_dist(rng);
        }
        rpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp.begin(), tmp.end()});
        cpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp2.begin(), tmp2.end()});
        rgather_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp3.begin(), tmp3.end()});
    }

    template <typename ConvertedType, typename InputType>
    std::unique_ptr<ConvertedType> convert(InputType&& input)
    {
        auto result = ConvertedType::create(input->get_executor());
        input->convert_to(result.get());
        return result;
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<ComplexMtx> c_y;
    std::unique_ptr<ComplexMtx> c_alpha;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> square;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<ComplexMtx> dc_y;
    std::unique_ptr<ComplexMtx> dc_alpha;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Mtx> dsquare;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Arr> rgather_idxs;
};


TEST_F(Dense, CopyRespectsStride)
{
    set_up_vector_data(3);
    auto stride = dx->get_size()[1] + 1;
    auto result = Mtx::create(exec, dx->get_size(), stride);
    vtype val = 1234567.0;
    auto original_data = result->get_values();
    auto padding_ptr = original_data + dx->get_size()[1];
    exec->copy_from(ref.get(), 1, &val, padding_ptr);

    dx->convert_to(result.get());

    GKO_ASSERT_MTX_NEAR(result, dx, 0);
    ASSERT_EQ(result->get_stride(), stride);
    ASSERT_EQ(exec->copy_val_to_host(padding_ptr), val);
    ASSERT_EQ(result->get_values(), original_data);
}


TEST_F(Dense, FillIsEquivalentToRef)
{
    set_up_vector_data(3);

    auto hand = dx->fill(42, this->exec->get_handle_at(0));
    x->fill(42);

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, 0);
}


TEST_F(Dense, StridedFillIsEquivalentToRef)
{
    using T = vtype;
    auto x = gko::initialize<gko::matrix::Dense<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, ref);
    auto dx = gko::initialize<gko::matrix::Dense<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, exec);

    x->fill(42);
    auto hand = dx->fill(42, this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, 0);
}


TEST_F(Dense, SingleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->scale(alpha.get());
    auto hand = dx->scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorInvScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->inv_scale(alpha.get());
    auto hand = dx->inv_scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorComplexScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->scale(c_alpha.get());
    auto hand = dc_x->scale(dc_alpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorComplexInvScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->inv_scale(c_alpha.get());
    auto hand = dc_x->inv_scale(dc_alpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorComplexRealScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->scale(alpha.get());
    auto hand = dc_x->scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorComplexRealInvScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->inv_scale(alpha.get());
    auto hand = dc_x->inv_scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    auto hand = dx->scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorInvScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->inv_scale(alpha.get());
    auto hand = dx->inv_scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->scale(c_alpha.get());
    auto hand = dc_x->scale(dc_alpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexInvScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->inv_scale(c_alpha.get());
    auto hand = dc_x->inv_scale(dc_alpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexRealScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->scale(alpha.get());
    auto hand = dc_x->scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexRealInvScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->inv_scale(alpha.get());
    auto hand = dc_x->inv_scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    auto hand = dx->scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorInvScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->inv_scale(alpha.get());
    auto hand = dx->inv_scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->scale(c_alpha.get());
    auto hand = dc_x->scale(dc_alpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexInvScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->inv_scale(c_alpha.get());
    auto hand = dc_x->inv_scale(dc_alpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexRealScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->scale(alpha.get());
    auto hand = dc_x->scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense,
       MultipleVectorComplexRealInvScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->inv_scale(alpha.get());
    auto hand = dc_x->inv_scale(dalpha.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    auto hand =
        dx->add_scaled(dalpha.get(), dy.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->sub_scaled(alpha.get(), y.get());
    auto hand =
        dx->sub_scaled(dalpha.get(), dy.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorComplexAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->add_scaled(c_alpha.get(), c_y.get());
    auto hand = dc_x->add_scaled(dc_alpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorComplexSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->sub_scaled(c_alpha.get(), c_y.get());
    auto hand = dc_x->sub_scaled(dc_alpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorComplexRealAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->add_scaled(alpha.get(), c_y.get());
    auto hand = dc_x->add_scaled(dalpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, SingleVectorComplexRealSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->sub_scaled(alpha.get(), c_y.get());
    auto hand = dc_x->sub_scaled(dalpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    auto hand =
        dx->add_scaled(dalpha.get(), dy.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->sub_scaled(alpha.get(), y.get());
    auto hand =
        dx->sub_scaled(dalpha.get(), dy.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->add_scaled(c_alpha.get(), c_y.get());
    auto hand = dc_x->add_scaled(dc_alpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->sub_scaled(c_alpha.get(), c_y.get());
    auto hand = dc_x->sub_scaled(dc_alpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexRealAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->add_scaled(alpha.get(), c_y.get());
    auto hand = dc_x->add_scaled(dalpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexRealSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->sub_scaled(alpha.get(), c_y.get());
    auto hand = dc_x->sub_scaled(dalpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->add_scaled(alpha.get(), y.get());
    auto hand =
        dx->add_scaled(dalpha.get(), dy.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorSubtractScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->sub_scaled(alpha.get(), y.get());
    auto hand =
        dx->sub_scaled(dalpha.get(), dy.get(), this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dx, x, r<vtype>::value);
}


TEST_F(Dense, MultipleVectorComplexAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->add_scaled(c_alpha.get(), c_y.get());
    auto hand = dc_x->add_scaled(dc_alpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense,
       MultipleVectorComplexSubtractScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->sub_scaled(c_alpha.get(), c_y.get());
    auto hand = dc_x->sub_scaled(dc_alpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(Dense,
       MultipleVectorComplexRealAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->add_scaled(alpha.get(), c_y.get());
    auto hand = dc_x->add_scaled(dalpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


TEST_F(
    Dense,
    MultipleVectorComplexRealSubtractScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->sub_scaled(alpha.get(), c_y.get());
    auto hand = dc_x->sub_scaled(dalpha.get(), dc_y.get(),
                                 this->exec->get_handle_at(0));

    hand->wait();
    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<vtype>::value);
}


}  // namespace
