// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/batch_multi_vector.hpp>


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>


#include "core/base/batch_multi_vector_kernels.hpp"
#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/executor.hpp"


class MultiVector : public CommonTestFixture {
protected:
    using Mtx = gko::batch::MultiVector<value_type>;
    using NormVector = gko::batch::MultiVector<gko::remove_complex<value_type>>;
    using ComplexMtx = gko::batch::MultiVector<std::complex<value_type>>;

    MultiVector() : rand_engine(15) {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(const gko::size_type num_batch_items,
                                     gko::size_type num_rows,
                                     gko::size_type num_cols)
    {
        return gko::test::generate_random_batch_matrix<MtxType>(
            num_batch_items, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_elem_scale_vector_data(gko::size_type num_vecs,
                                       const int num_rows = 252)
    {
        x = gen_mtx<Mtx>(batch_size, num_rows, num_vecs);
        alpha = gen_mtx<Mtx>(batch_size, num_rows, num_vecs);
        dx = gko::clone(exec, x);
        dalpha = gko::clone(exec, alpha);
    }

    void set_up_vector_data(gko::size_type num_vecs, const int num_rows = 252,
                            bool different_alpha = false)
    {
        x = gen_mtx<Mtx>(batch_size, num_rows, num_vecs);
        y = gen_mtx<Mtx>(batch_size, num_rows, num_vecs);
        c_x = gen_mtx<ComplexMtx>(batch_size, num_rows, num_vecs);
        c_y = gen_mtx<ComplexMtx>(batch_size, num_rows, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(batch_size, 1, num_vecs);
            beta = gen_mtx<Mtx>(batch_size, 1, num_vecs);
        } else {
            alpha = gko::batch::initialize<Mtx>(batch_size, {2.0}, ref);
            beta = gko::batch::initialize<Mtx>(batch_size, {-0.5}, ref);
        }
        dx = gko::clone(exec, x);
        dy = gko::clone(exec, y);
        dc_x = gko::clone(exec, c_x);
        dc_y = gko::clone(exec, c_y);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
        expected = Mtx::create(
            ref, gko::batch_dim<2>(batch_size, gko::dim<2>{1, num_vecs}));
        dresult = Mtx::create(
            exec, gko::batch_dim<2>(batch_size, gko::dim<2>{1, num_vecs}));
    }

    std::default_random_engine rand_engine;

    const size_t batch_size = 11;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<ComplexMtx> c_y;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> square;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<ComplexMtx> dc_y;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Mtx> dsquare;
};


TEST_F(MultiVector, SingleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 5 * r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, 252, true);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 5 * r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 5 * r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 5 * r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 5 * r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorElemWiseScaleIsEquivalentToRef)
{
    set_up_elem_scale_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 5 * r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm2SingleSmallIsEquivalentToRef)
{
    set_up_vector_data(1, 10);
    auto norm_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_common_size()[1]});
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->exec, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_BATCH_MTX_NEAR(norm_expected, dnorm, 5 * r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm2SingleIsEquivalentToRef)
{
    set_up_vector_data(1);
    auto norm_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_common_size()[1]});
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->exec, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_BATCH_MTX_NEAR(norm_expected, dnorm, 5 * r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);
    auto norm_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_common_size()[1]});
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->exec, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_BATCH_MTX_NEAR(norm_expected, dnorm, 5 * r<value_type>::value);
}


TEST_F(MultiVector, ComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);
    auto dot_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_common_size()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->exec, dot_size);
    auto cdot_expected = ComplexMtx::create(this->ref, dot_size);
    auto dc_dot = ComplexMtx::create(this->exec, dot_size);

    x->compute_dot(y.get(), dot_expected.get());
    dx->compute_dot(dy.get(), ddot.get());
    c_x->compute_dot(c_y.get(), cdot_expected.get());
    dc_x->compute_dot(dc_y.get(), dc_dot.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 5 * r<value_type>::value);
    GKO_ASSERT_BATCH_MTX_NEAR(cdot_expected, dc_dot, 5 * r<value_type>::value);
}


TEST_F(MultiVector, ComputeDotSingleIsEquivalentToRef)
{
    set_up_vector_data(1);
    auto dot_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_common_size()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->exec, dot_size);

    x->compute_dot(y.get(), dot_expected.get());
    dx->compute_dot(dy.get(), ddot.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 5 * r<value_type>::value);
}


TEST_F(MultiVector, ComputeDotSingleSmallIsEquivalentToRef)
{
    set_up_vector_data(1, 10);
    auto dot_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_common_size()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->exec, dot_size);

    x->compute_dot(y.get(), dot_expected.get());
    dx->compute_dot(dy.get(), ddot.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 5 * r<value_type>::value);
}


TEST_F(MultiVector, ComputeConjDotIsEquivalentToRef)
{
    set_up_vector_data(20);
    auto dot_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_common_size()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->exec, dot_size);
    auto cdot_expected = ComplexMtx::create(this->ref, dot_size);
    auto dc_dot = ComplexMtx::create(this->exec, dot_size);

    x->compute_conj_dot(y.get(), dot_expected.get());
    dx->compute_conj_dot(dy.get(), ddot.get());
    c_x->compute_conj_dot(c_y.get(), cdot_expected.get());
    dc_x->compute_conj_dot(dc_y.get(), dc_dot.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 5 * r<value_type>::value);
    GKO_ASSERT_BATCH_MTX_NEAR(cdot_expected, dc_dot, 5 * r<value_type>::value);
}


TEST_F(MultiVector, ComputeConjDotSingleIsEquivalentToRef)
{
    set_up_vector_data(1);
    auto dot_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_common_size()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->exec, dot_size);

    x->compute_conj_dot(y.get(), dot_expected.get());
    dx->compute_conj_dot(dy.get(), ddot.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 5 * r<value_type>::value);
}


TEST_F(MultiVector, CopySingleIsEquivalentToRef)
{
    set_up_vector_data(1);

    gko::kernels::reference::batch_multi_vector::copy(this->ref, x.get(),
                                                      y.get());
    gko::kernels::EXEC_NAMESPACE::batch_multi_vector::copy(this->exec, dx.get(),
                                                           dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dy, y, 0.0);
}


TEST_F(MultiVector, CopyIsEquivalentToRef)
{
    set_up_vector_data(20);

    gko::kernels::reference::batch_multi_vector::copy(this->ref, x.get(),
                                                      y.get());
    gko::kernels::EXEC_NAMESPACE::batch_multi_vector::copy(this->exec, dx.get(),
                                                           dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dy, y, 0.0);
}
