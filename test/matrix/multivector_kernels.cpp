// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/multivector_kernels.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/multivector.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>

#include "core/components/fill_array_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class MultiVector : public CommonTestFixture {
protected:
    // in single mode, mixed_type will be the same as value_type
    using mixed_type = float;
    using Mtx = gko::matrix::MultiVector<value_type>;
    using MixedMtx = gko::matrix::MultiVector<mixed_type>;
    using NormVector =
        gko::matrix::MultiVector<gko::remove_complex<value_type>>;
    using Arr = gko::array<index_type>;
    using ComplexMtx = gko::matrix::MultiVector<std::complex<value_type>>;
    using Diagonal = gko::matrix::Diagonal<value_type>;
    using Permutation = gko::matrix::Permutation<index_type>;
    using ScaledPermutation =
        gko::matrix::ScaledPermutation<value_type, index_type>;

    MultiVector() : rand_engine(15) {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<gko::remove_complex<value_type>>(0.0, 1.0),
            rand_engine, ref);
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<gko::remove_complex<value_type>>(-1.0,
                                                                      1.0),
            rand_engine, ref);
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
            c_alpha = gko::initialize<ComplexMtx>(
                {std::complex<value_type>{2.0}}, ref);
        }
        dx = gko::clone(exec, x);
        dy = gko::clone(exec, y);
        dc_x = gko::clone(exec, c_x);
        dc_y = gko::clone(exec, c_y);
        dalpha = gko::clone(exec, alpha);
        dc_alpha = gko::clone(exec, c_alpha);
        result = Mtx::create(ref, gko::dim<2>{1, num_vecs});
        dresult = Mtx::create(exec, gko::dim<2>{1, num_vecs});
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(65, 25);
        y = gen_mtx<Mtx>(25, 35);
        c_x = gen_mtx<ComplexMtx>(65, 25);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        result = gen_mtx<Mtx>(65, 35);
        square = gen_mtx<Mtx>(x->get_size()[0], x->get_size()[0]);
        dx = gko::clone(exec, x);
        dy = gko::clone(exec, y);
        dc_x = gko::clone(exec, c_x);
        dresult = gko::clone(exec, result);
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
        std::vector<value_type> scale_factors(tmp.size());
        std::vector<value_type> scale_factors2(tmp2.size());
        std::uniform_int_distribution<int> row_dist(0, x->get_size()[0] - 1);
        std::uniform_real_distribution<value_type> scale_dist{1, 2};
        for (auto& i : tmp3) {
            i = row_dist(rng);
        }
        for (auto& s : scale_factors) {
            s = scale_dist(rng);
        }
        for (auto& s : scale_factors2) {
            s = scale_dist(rng);
        }
        rpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp.begin(), tmp.end()});
        cpermute_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp2.begin(), tmp2.end()});
        rgather_idxs =
            std::unique_ptr<Arr>(new Arr{ref, tmp3.begin(), tmp3.end()});
        rpermutation = Permutation::create(ref, *rpermute_idxs);
        cpermutation = Permutation::create(ref, *cpermute_idxs);
        rspermutation = ScaledPermutation::create(
            ref,
            gko::array<value_type>{ref, scale_factors.begin(),
                                   scale_factors.end()},
            *rpermute_idxs);
        cspermutation = ScaledPermutation::create(
            ref,
            gko::array<value_type>{ref, scale_factors2.begin(),
                                   scale_factors2.end()},
            *cpermute_idxs);
    }

    template <typename ConvertedType, typename InputType>
    std::unique_ptr<ConvertedType> convert(InputType&& input)
    {
        auto result = ConvertedType::create(input->get_executor());
        input->convert_to(result);
        return result;
    }

    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<ComplexMtx> c_y;
    std::unique_ptr<ComplexMtx> c_alpha;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> result;
    std::unique_ptr<Mtx> square;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<ComplexMtx> dc_y;
    std::unique_ptr<ComplexMtx> dc_alpha;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dsquare;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
    std::unique_ptr<Permutation> rpermutation;
    std::unique_ptr<Permutation> cpermutation;
    std::unique_ptr<ScaledPermutation> rspermutation;
    std::unique_ptr<ScaledPermutation> cspermutation;
    std::unique_ptr<Arr> rgather_idxs;
};


TEST_F(MultiVector, SingleVectorComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_dot(y, result);
    dx->compute_dot(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, result, 5 * r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_dot(y, result);
    dx->compute_dot(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, result, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComputeConjDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_conj_dot(y, result);
    dx->compute_conj_dot(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, result, 5 * r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComputeConjDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_conj_dot(y, result);
    dx->compute_conj_dot(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, result, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(1);
    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->exec, norm_size);

    x->compute_norm2(norm_expected);
    dx->compute_norm2(dnorm);

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);
    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->exec, norm_size);

    x->compute_norm2(norm_expected);
    dx->compute_norm2(dnorm);

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<value_type>::value);
}


TEST_F(MultiVector, ComputeDotComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_x = gko::clone(exec, complex_x);
    auto result = ComplexMtx::create(ref, gko::dim<2>{1, 2});
    auto dresult = ComplexMtx::create(exec, gko::dim<2>{1, 2});

    complex_b->compute_dot(complex_x, result);
    dcomplex_b->compute_dot(dcomplex_x, dresult);

    GKO_ASSERT_MTX_NEAR(result, dresult, r<value_type>::value * 2);
}


TEST_F(MultiVector, ComputeConjDotComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexMtx>(1234, 2);
    auto dcomplex_x = gko::clone(exec, complex_x);
    auto result = ComplexMtx::create(ref, gko::dim<2>{1, 2});
    auto dresult = ComplexMtx::create(exec, gko::dim<2>{1, 2});

    complex_b->compute_conj_dot(complex_x, result);
    dcomplex_b->compute_conj_dot(dcomplex_x, dresult);

    GKO_ASSERT_MTX_NEAR(result, dresult, r<value_type>::value * 2);
}


TEST_F(MultiVector, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(dtrans.get()),
                        static_cast<Mtx*>(trans.get()), 0);
}


TEST_F(MultiVector, IsTransposableIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::local_span{0, x->get_size()[0] - 2};
    auto col_span = gko::local_span{0, x->get_size()[1] - 2};
    auto sub_x = x->create_subview(row_span, col_span);
    auto sub_dx = dx->create_subview(row_span, col_span);
    // create the target matrices on another executor to
    // force temporary clone
    auto trans = Mtx::create(ref, gko::transpose(sub_x->get_size()));
    auto dtrans = Mtx::create(ref, gko::transpose(sub_x->get_size()),
                              sub_x->get_size()[0] + 4);

    sub_x->transpose(trans);
    sub_dx->transpose(dtrans);

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0);
}


TEST_F(MultiVector, IsConjugateTransposable)
{
    set_up_apply_data();

    auto trans = c_x->conj_transpose();
    auto dtrans = dc_x->conj_transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx*>(dtrans.get()),
                        static_cast<ComplexMtx*>(trans.get()), 0);
}


TEST_F(MultiVector, IsConjugateTransposableIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::local_span{0, c_x->get_size()[0] - 2};
    auto col_span = gko::local_span{0, c_x->get_size()[1] - 2};
    auto sub_x = c_x->create_subview(row_span, col_span);
    auto sub_dx = dc_x->create_subview(row_span, col_span);
    // create the target matrices on another executor to
    // force temporary clone
    auto trans = ComplexMtx::create(ref, gko::transpose(sub_x->get_size()));
    auto dtrans = ComplexMtx::create(ref, gko::transpose(sub_x->get_size()),
                                     sub_x->get_size()[0] + 4);

    sub_x->conj_transpose(trans);
    sub_dx->conj_transpose(dtrans);

    GKO_ASSERT_MTX_NEAR(dtrans, trans, 0);
}


TEST_F(MultiVector, CopyRespectsStride)
{
    set_up_vector_data(3);
    auto stride = dx->get_size()[1] + 1;
    auto result = Mtx::create(exec, dx->get_size(), stride);
    value_type val = 1234567.0;
    auto original_data = result->get_values();
    auto padding_ptr = original_data + dx->get_size()[1];
    exec->copy_from(ref, 1, &val, padding_ptr);

    dx->convert_to(result);

    GKO_ASSERT_MTX_NEAR(result, dx, 0);
    ASSERT_EQ(result->get_stride(), stride);
    ASSERT_EQ(exec->copy_val_to_host(padding_ptr), val);
    ASSERT_EQ(result->get_values(), original_data);
}


TEST_F(MultiVector, FillIsEquivalentToRef)
{
    set_up_vector_data(3);

    x->fill(42);
    dx->fill(42);

    GKO_ASSERT_MTX_NEAR(dx, x, 0);
}


TEST_F(MultiVector, StridedFillIsEquivalentToRef)
{
    using T = value_type;
    auto x = gko::initialize<gko::matrix::MultiVector<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, ref);
    auto dx = gko::initialize<gko::matrix::MultiVector<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, exec);

    x->fill(42);
    dx->fill(42);

    GKO_ASSERT_MTX_NEAR(dx, x, 0);
}


TEST_F(MultiVector, SingleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->scale(alpha);
    dx->scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorInvScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->inv_scale(alpha);
    dx->inv_scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComplexScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->scale(c_alpha);
    dc_x->scale(dc_alpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComplexInvScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->inv_scale(c_alpha);
    dc_x->inv_scale(dc_alpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComplexRealScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->scale(alpha);
    dc_x->scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComplexRealInvScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->inv_scale(alpha);
    dc_x->inv_scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha);
    dx->scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorInvScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->inv_scale(alpha);
    dx->inv_scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComplexScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->scale(c_alpha);
    dc_x->scale(dc_alpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComplexInvScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->inv_scale(c_alpha);
    dc_x->inv_scale(dc_alpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComplexRealScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->scale(alpha);
    dc_x->scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComplexRealInvScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->inv_scale(alpha);
    dc_x->inv_scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha);
    dx->scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorInvScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->inv_scale(alpha);
    dx->inv_scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector,
       MultipleVectorComplexScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->scale(c_alpha);
    dc_x->scale(dc_alpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector,
       MultipleVectorComplexInvScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->inv_scale(c_alpha);
    dc_x->inv_scale(dc_alpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector,
       MultipleVectorComplexRealScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->scale(alpha);
    dc_x->scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector,
       MultipleVectorComplexRealInvScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->inv_scale(alpha);
    dc_x->inv_scale(dalpha);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha, y);
    dx->add_scaled(dalpha, dy);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->sub_scaled(alpha, y);
    dx->sub_scaled(dalpha, dy);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComplexAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->add_scaled(c_alpha, c_y);
    dc_x->add_scaled(dc_alpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComplexSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->sub_scaled(c_alpha, c_y);
    dc_x->sub_scaled(dc_alpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComplexRealAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->add_scaled(alpha, c_y);
    dc_x->add_scaled(dalpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, SingleVectorComplexRealSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    c_x->sub_scaled(alpha, c_y);
    dc_x->sub_scaled(dalpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha, y);
    dx->add_scaled(dalpha, dy);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->sub_scaled(alpha, y);
    dx->sub_scaled(dalpha, dy);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComplexAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->add_scaled(c_alpha, c_y);
    dc_x->add_scaled(dc_alpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComplexSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->sub_scaled(c_alpha, c_y);
    dc_x->sub_scaled(dc_alpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComplexRealAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->add_scaled(alpha, c_y);
    dc_x->add_scaled(dalpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorComplexRealSubtractScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    c_x->sub_scaled(alpha, c_y);
    dc_x->sub_scaled(dalpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, MultipleVectorAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->add_scaled(alpha, y);
    dx->add_scaled(dalpha, dy);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector,
       MultipleVectorSubtractScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->sub_scaled(alpha, y);
    dx->sub_scaled(dalpha, dy);

    GKO_ASSERT_MTX_NEAR(dx, x, r<value_type>::value);
}


TEST_F(MultiVector,
       MultipleVectorComplexAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->add_scaled(c_alpha, c_y);
    dc_x->add_scaled(dc_alpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector,
       MultipleVectorComplexSubtractScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->sub_scaled(c_alpha, c_y);
    dc_x->sub_scaled(dc_alpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector,
       MultipleVectorComplexRealAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->add_scaled(alpha, c_y);
    dc_x->add_scaled(dalpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(
    MultiVector,
    MultipleVectorComplexRealSubtractScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    c_x->sub_scaled(alpha, c_y);
    dc_x->sub_scaled(dalpha, dc_y);

    GKO_ASSERT_MTX_NEAR(dc_x, c_x, r<value_type>::value);
}


TEST_F(MultiVector, CanGatherRows)
{
    set_up_apply_data();

    auto r_gather = x->row_gather(rgather_idxs.get());
    auto dr_gather = dx->row_gather(rgather_idxs.get());

    GKO_ASSERT_MTX_NEAR(r_gather, dr_gather, 0);
}


TEST_F(MultiVector, CanGatherRowsIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::local_span{0, x->get_size()[0]};
    auto col_span = gko::local_span{0, x->get_size()[1] - 2};
    auto sub_x = x->create_subview(row_span, col_span);
    auto sub_dx = dx->create_subview(row_span, col_span);
    auto gather_size =
        gko::dim<2>{rgather_idxs->get_size(), sub_x->get_size()[1]};
    auto r_gather = Mtx::create(ref, gather_size);
    // test make_temporary_clone and non-default stride
    auto dr_gather = Mtx::create(ref, gather_size, sub_x->get_size()[1] + 2);

    sub_x->row_gather(rgather_idxs.get(), r_gather);
    sub_dx->row_gather(rgather_idxs.get(), dr_gather);

    GKO_ASSERT_MTX_NEAR(r_gather, dr_gather, 0);
}


TEST_F(MultiVector, CanAdvancedGatherRowsIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::local_span{0, x->get_size()[0]};
    auto col_span = gko::local_span{0, x->get_size()[1] - 2};
    auto sub_x = x->create_subview(row_span, col_span);
    auto sub_dx = dx->create_subview(row_span, col_span);
    auto gather_size =
        gko::dim<2>{rgather_idxs->get_size(), sub_x->get_size()[1]};
    auto r_gather = gen_mtx<Mtx>(gather_size[0], gather_size[1]);
    // test make_temporary_clone and non-default stride
    auto dr_gather = Mtx::create(ref, gather_size, sub_x->get_size()[1] + 2);
    dr_gather->copy_from(r_gather);

    sub_x->row_gather(alpha, rgather_idxs.get(), beta, r_gather);
    sub_dx->row_gather(dalpha, rgather_idxs.get(), dbeta, dr_gather);

    GKO_ASSERT_MTX_NEAR(r_gather, dr_gather, 0);
}


TEST_F(MultiVector, CanGatherRowsIntoMixedMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::local_span{0, x->get_size()[0]};
    auto col_span = gko::local_span{0, x->get_size()[1] - 2};
    auto sub_x = x->create_subview(row_span, col_span);
    auto sub_dx = dx->create_subview(row_span, col_span);
    auto gather_size =
        gko::dim<2>{rgather_idxs->get_size(), sub_x->get_size()[1]};
    auto r_gather = MixedMtx::create(ref, gather_size);
    // test make_temporary_clone and non-default stride
    auto dr_gather =
        MixedMtx::create(ref, gather_size, sub_x->get_size()[1] + 2);

    sub_x->row_gather(rgather_idxs.get(), r_gather);
    sub_dx->row_gather(rgather_idxs.get(), dr_gather);

    GKO_ASSERT_MTX_NEAR(r_gather, dr_gather, 0);
}


TEST_F(MultiVector, CanAdvancedGatherRowsIntoMixedMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto row_span = gko::local_span{0, x->get_size()[0]};
    auto col_span = gko::local_span{0, x->get_size()[1] - 2};
    auto sub_x = x->create_subview(row_span, col_span);
    auto sub_dx = dx->create_subview(row_span, col_span);
    auto gather_size =
        gko::dim<2>{rgather_idxs->get_size(), sub_x->get_size()[1]};
    auto r_gather = gen_mtx<MixedMtx>(gather_size[0], gather_size[1]);
    // test make_temporary_clone and non-default stride
    auto dr_gather =
        MixedMtx::create(ref, gather_size, sub_x->get_size()[1] + 2);
    dr_gather->copy_from(r_gather);

    sub_x->row_gather(alpha, rgather_idxs.get(), beta, r_gather);
    sub_dx->row_gather(alpha, rgather_idxs.get(), beta, dr_gather);

    GKO_ASSERT_MTX_NEAR(r_gather, dr_gather, 0);
}


TEST_F(MultiVector, IsGenericPermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto permuted = square->permute(rpermutation, mode);
        auto dpermuted = dsquare->permute(rpermutation, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
    }
}


TEST_F(MultiVector, IsGenericPermutableRectangular)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::inverse_rows,
          permute_mode::inverse_columns}) {
        SCOPED_TRACE(mode);
        auto perm = (mode & permute_mode::rows) == permute_mode::rows
                        ? rpermutation.get()
                        : cpermutation.get();

        auto permuted = x->permute(perm, mode);
        auto dpermuted = dx->permute(perm, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
    }
}


TEST_F(MultiVector, IsGenericPermutableIntoMultiVectorCrossExecutor)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto host_permuted = square->clone();

        auto ref_permuted = square->permute(rpermutation, mode);
        dsquare->permute(rpermutation, host_permuted, mode);

        GKO_ASSERT_MTX_NEAR(ref_permuted, host_permuted, 0);
    }
}


TEST_F(MultiVector, IsNonsymmPermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto permuted = x->permute(rpermutation, cpermutation, invert);
        auto dpermuted = dx->permute(rpermutation, cpermutation, invert);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
    }
}


TEST_F(MultiVector, IsNonsymmPermutableIntoMultiVectorCrossExecutor)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto host_permuted = dx->clone();

        auto ref_permuted = x->permute(rpermutation, cpermutation, invert);
        dx->permute(rpermutation, cpermutation, host_permuted, invert);

        GKO_ASSERT_MTX_NEAR(ref_permuted, host_permuted, 0);
    }
}


TEST_F(MultiVector, IsGenericScalePermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto permuted = square->scale_permute(rspermutation, mode);
        auto dpermuted = dsquare->scale_permute(rspermutation, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, r<value_type>::value);
    }
}


TEST_F(MultiVector, IsGenericScalePermutableRectangular)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto mode :
         {permute_mode::rows, permute_mode::columns, permute_mode::inverse_rows,
          permute_mode::inverse_columns}) {
        SCOPED_TRACE(mode);
        auto perm = (mode & permute_mode::rows) == permute_mode::rows
                        ? rspermutation.get()
                        : cspermutation.get();

        auto permuted = x->scale_permute(perm, mode);
        auto dpermuted = dx->scale_permute(perm, mode);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, r<value_type>::value);
    }
}


TEST_F(MultiVector, IsGenericScalePermutableIntoMultiVectorCrossExecutor)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto mode :
         {permute_mode::none, permute_mode::rows, permute_mode::columns,
          permute_mode::symmetric, permute_mode::inverse_rows,
          permute_mode::inverse_columns, permute_mode::inverse_symmetric}) {
        SCOPED_TRACE(mode);
        auto host_permuted = square->clone();

        auto ref_permuted = square->scale_permute(rspermutation, mode);
        dsquare->scale_permute(rspermutation, host_permuted, mode);

        GKO_ASSERT_MTX_NEAR(ref_permuted, host_permuted, r<value_type>::value);
    }
}


TEST_F(MultiVector, IsNonsymmScalePermutable)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto permuted = x->scale_permute(rspermutation, cspermutation, invert);
        auto dpermuted =
            dx->scale_permute(rspermutation, cspermutation, invert);

        GKO_ASSERT_MTX_NEAR(permuted, dpermuted, r<value_type>::value);
    }
}


TEST_F(MultiVector, IsNonsymmScalePermutableIntoMultiVectorCrossExecutor)
{
    using gko::matrix::permute_mode;
    set_up_apply_data();

    for (auto invert : {false, true}) {
        SCOPED_TRACE(invert);
        auto host_permuted = dx->clone();

        auto ref_permuted =
            x->scale_permute(rspermutation, cspermutation, invert);
        dx->scale_permute(rspermutation, cspermutation, host_permuted, invert);

        GKO_ASSERT_MTX_NEAR(ref_permuted, host_permuted, r<value_type>::value);
    }
}


TEST_F(MultiVector, IsPermutable)
{
    set_up_apply_data();

    auto permuted = square->permute(rpermute_idxs.get());
    auto dpermuted = dsquare->permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(permuted.get()),
                        static_cast<Mtx*>(dpermuted.get()), 0);
}


TEST_F(MultiVector, IsPermutableIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, square->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted =
        Mtx::create(ref, square->get_size(), square->get_size()[1] + 2);

    square->permute(rpermute_idxs.get(), permuted);
    dsquare->permute(rpermute_idxs.get(), dpermuted);

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(MultiVector, IsInversePermutable)
{
    set_up_apply_data();

    auto permuted = square->inverse_permute(rpermute_idxs.get());
    auto dpermuted = dsquare->inverse_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(permuted.get()),
                        static_cast<Mtx*>(dpermuted.get()), 0);
}


TEST_F(MultiVector, IsInversePermutableIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, square->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted =
        Mtx::create(ref, square->get_size(), square->get_size()[1] + 2);

    square->inverse_permute(rpermute_idxs.get(), permuted);
    dsquare->inverse_permute(rpermute_idxs.get(), dpermuted);

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(MultiVector, IsRowPermutable)
{
    set_up_apply_data();

    auto r_permute = x->row_permute(rpermute_idxs.get());
    auto dr_permute = dx->row_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(r_permute.get()),
                        static_cast<Mtx*>(dr_permute.get()), 0);
}


TEST_F(MultiVector, IsRowPermutableIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->row_permute(rpermute_idxs.get(), permuted);
    dx->row_permute(rpermute_idxs.get(), dpermuted);

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(MultiVector, IsColPermutable)
{
    set_up_apply_data();

    auto c_permute = x->column_permute(cpermute_idxs.get());
    auto dc_permute = dx->column_permute(cpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(c_permute.get()),
                        static_cast<Mtx*>(dc_permute.get()), 0);
}


TEST_F(MultiVector, IsColPermutableIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->column_permute(cpermute_idxs.get(), permuted);
    dx->column_permute(cpermute_idxs.get(), dpermuted);

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(MultiVector, IsInverseRowPermutable)
{
    set_up_apply_data();

    auto inverse_r_permute = x->inverse_row_permute(rpermute_idxs.get());
    auto d_inverse_r_permute = dx->inverse_row_permute(rpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(inverse_r_permute.get()),
                        static_cast<Mtx*>(d_inverse_r_permute.get()), 0);
}


TEST_F(MultiVector, IsInverseRowPermutableIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->inverse_row_permute(rpermute_idxs.get(), permuted);
    dx->inverse_row_permute(rpermute_idxs.get(), dpermuted);

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(MultiVector, IsInverseColPermutable)
{
    set_up_apply_data();

    auto inverse_c_permute = x->inverse_column_permute(cpermute_idxs.get());
    auto d_inverse_c_permute = dx->inverse_column_permute(cpermute_idxs.get());

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(inverse_c_permute.get()),
                        static_cast<Mtx*>(d_inverse_c_permute.get()), 0);
}


TEST_F(MultiVector, IsInverseColPermutableIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto permuted = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dpermuted = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->inverse_column_permute(cpermute_idxs.get(), permuted);
    dx->inverse_column_permute(cpermute_idxs.get(), dpermuted);

    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(MultiVector, ComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    auto dot_size = gko::dim<2>{1, x->get_size()[1]};
    auto dot_expected = Mtx::create(ref, dot_size);
    auto ddot = Mtx::create(ref, dot_size);

    // all parameters are on ref to check cross-executor calls
    x->compute_dot(y, dot_expected);
    dx->compute_dot(y, ddot);

    GKO_ASSERT_MTX_NEAR(ddot, dot_expected, r<value_type>::value * 5);
}


TEST_F(MultiVector, ComputeDotWithPreallocatedTmpIsEquivalentToRef)
{
    set_up_vector_data(42);

    auto dot_size = gko::dim<2>{1, x->get_size()[1]};
    auto dot_expected = Mtx::create(ref, dot_size);
    auto ddot = Mtx::create(ref, dot_size);
    gko::array<char> tmp{exec, 12345};

    // all parameters are on ref to check cross-executor calls
    x->compute_dot(y, dot_expected);
    dx->compute_dot(y, ddot, tmp);

    GKO_ASSERT_MTX_NEAR(ddot, dot_expected, r<value_type>::value);
}


TEST_F(MultiVector, ComputeDotWithTmpIsEquivalentToRef)
{
    set_up_vector_data(40);

    auto dot_size = gko::dim<2>{1, x->get_size()[1]};
    auto dot_expected = Mtx::create(ref, dot_size);
    auto ddot = Mtx::create(ref, dot_size);
    gko::array<char> tmp{exec};

    // all parameters are on ref to check cross-executor calls
    x->compute_dot(y, dot_expected);
    dx->compute_dot(y, ddot, tmp);

    GKO_ASSERT_MTX_NEAR(ddot, dot_expected, r<value_type>::value);
}


TEST_F(MultiVector, ComputeConjDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    auto dot_size = gko::dim<2>{1, x->get_size()[1]};
    auto dot_expected = Mtx::create(ref, dot_size);
    auto ddot = Mtx::create(ref, dot_size);

    // all parameters are on ref to check cross-executor calls
    x->compute_conj_dot(y, dot_expected);
    dx->compute_conj_dot(y, ddot);

    GKO_ASSERT_MTX_NEAR(ddot, dot_expected, r<value_type>::value * 5);
}


TEST_F(MultiVector, ComputeConjDotWithPreallocatedTmpIsEquivalentToRef)
{
    set_up_vector_data(36);

    auto dot_size = gko::dim<2>{1, x->get_size()[1]};
    auto dot_expected = Mtx::create(ref, dot_size);
    auto ddot = Mtx::create(ref, dot_size);
    gko::array<char> tmp{exec, 12345};

    // all parameters are on ref to check cross-executor calls
    x->compute_conj_dot(y, dot_expected);
    dx->compute_conj_dot(y, ddot, tmp);

    GKO_ASSERT_MTX_NEAR(ddot, dot_expected, r<value_type>::value);
}


TEST_F(MultiVector, ComputeConjDotWithTmpIsEquivalentToRef)
{
    set_up_vector_data(65);

    auto dot_size = gko::dim<2>{1, x->get_size()[1]};
    auto dot_expected = Mtx::create(ref, dot_size);
    auto ddot = Mtx::create(ref, dot_size);
    gko::array<char> tmp{ref};

    // all parameters are on ref to check cross-executor calls
    x->compute_conj_dot(y, dot_expected);
    dx->compute_conj_dot(y, ddot, tmp);

    GKO_ASSERT_MTX_NEAR(ddot, dot_expected, r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm1IsEquivalentToRef)
{
    set_up_vector_data(2);

    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(ref, norm_size);
    auto dnorm = NormVector::create(exec, norm_size);

    // all parameters are on ref to check cross-executor calls
    x->compute_norm1(norm_expected);
    dx->compute_norm1(dnorm);

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm1WithPreallocatedTmpIsEquivalentToRef)
{
    set_up_vector_data(7);

    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(ref, norm_size);
    auto dnorm = NormVector::create(ref, norm_size);
    gko::array<char> tmp{exec, 12345};

    // all parameters are on ref to check cross-executor calls
    x->compute_norm1(norm_expected);
    dx->compute_norm1(dnorm, tmp);

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm1WithTmpIsEquivalentToRef)
{
    set_up_vector_data(10);

    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(ref, norm_size);
    auto dnorm = NormVector::create(ref, norm_size);
    gko::array<char> tmp{ref};

    // all parameters are on ref to check cross-executor calls
    x->compute_norm1(norm_expected);
    dx->compute_norm1(dnorm, tmp);

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(1);

    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(ref, norm_size);
    auto dnorm = NormVector::create(ref, norm_size);

    // all parameters are on ref to check cross-executor calls
    x->compute_norm1(norm_expected);
    dx->compute_norm1(dnorm);

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm2WithPreallocatedTmpIsEquivalentToRef)
{
    set_up_vector_data(3);

    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(ref, norm_size);
    auto dnorm = NormVector::create(ref, norm_size);
    gko::array<char> tmp{ref};

    // all parameters are on ref to check cross-executor calls
    x->compute_norm1(norm_expected);
    dx->compute_norm1(dnorm, tmp);

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<value_type>::value);
}


TEST_F(MultiVector, ComputeNorm2WithTmpIsEquivalentToRef)
{
    set_up_vector_data(14);

    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(ref, norm_size);
    auto dnorm = NormVector::create(ref, norm_size);
    gko::array<char> tmp{exec, 12345};

    // all parameters are on ref to check cross-executor calls
    x->compute_norm1(norm_expected);
    dx->compute_norm1(dnorm, tmp);

    GKO_ASSERT_MTX_NEAR(norm_expected, dnorm, r<value_type>::value);
}


TEST_F(MultiVector, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    x->compute_absolute_inplace();
    dx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(x, dx, r<value_type>::value);
}


TEST_F(MultiVector, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    auto abs_x = x->compute_absolute();
    auto dabs_x = dx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_x, dabs_x, r<value_type>::value);
}


TEST_F(MultiVector, OutplaceAbsoluteMatrixIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto abs_x = NormVector::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dabs_x = NormVector::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->compute_absolute(abs_x);
    dx->compute_absolute(dabs_x);

    GKO_ASSERT_MTX_NEAR(abs_x, dabs_x, r<value_type>::value);
}


TEST_F(MultiVector, MakeComplexIsEquivalentToRef)
{
    set_up_apply_data();

    auto complex_x = x->make_complex();
    auto dcomplex_x = dx->make_complex();

    GKO_ASSERT_MTX_NEAR(complex_x, dcomplex_x, 0);
}


TEST_F(MultiVector, MakeComplexIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto complex_x = ComplexMtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dcomplex_x =
        ComplexMtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->make_complex(complex_x);
    dx->make_complex(dcomplex_x);

    GKO_ASSERT_MTX_NEAR(complex_x, dcomplex_x, 0);
}


TEST_F(MultiVector, GetRealIsEquivalentToRef)
{
    set_up_apply_data();

    auto real_x = x->get_real();
    auto dreal_x = dx->get_real();

    GKO_ASSERT_MTX_NEAR(real_x, dreal_x, 0);
}


TEST_F(MultiVector, GetRealIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto real_x = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dreal_x = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->get_real(real_x);
    dx->get_real(dreal_x);

    GKO_ASSERT_MTX_NEAR(real_x, dreal_x, 0);
}


TEST_F(MultiVector, GetImagIsEquivalentToRef)
{
    set_up_apply_data();

    auto imag_x = x->get_imag();
    auto dimag_x = dx->get_imag();

    GKO_ASSERT_MTX_NEAR(imag_x, dimag_x, 0);
}


TEST_F(MultiVector, GetImagIntoMultiVectorCrossExecutor)
{
    set_up_apply_data();
    auto imag_x = Mtx::create(ref, x->get_size());
    // test make_temporary_clone and non-default stride
    auto dimag_x = Mtx::create(ref, x->get_size(), x->get_size()[1] + 2);

    x->get_imag(imag_x);
    dx->get_imag(dimag_x);

    GKO_ASSERT_MTX_NEAR(imag_x, dimag_x, 0);
}


TEST_F(MultiVector, ComputeNorm2SquaredIsEquivalentToRef)
{
    set_up_apply_data();
    auto norm_size = gko::dim<2>{1, x->get_size()[1]};
    auto norm_expected = NormVector::create(ref, norm_size);
    auto dnorm = NormVector::create(exec, norm_size);
    gko::array<char> tmp{ref};
    gko::array<char> dtmp{exec};

    gko::kernels::reference::multivector::compute_squared_norm2(
        ref, x->get_const_device_view(), norm_expected->get_device_view(), tmp);
    gko::kernels::GKO_DEVICE_NAMESPACE::multivector::compute_squared_norm2(
        exec, dx->get_const_device_view(), dnorm->get_device_view(), dtmp);

    GKO_ASSERT_MTX_NEAR(dnorm, norm_expected, r<value_type>::value);
}


TEST_F(MultiVector, ComputesSqrt)
{
    auto mtx = gko::test::generate_random_matrix<NormVector>(
        1, 7, std::uniform_int_distribution<int>(7, 7),
        std::uniform_real_distribution<gko::remove_complex<value_type>>(0, 10),
        rand_engine, ref);
    auto dmtx = gko::clone(exec, mtx);

    gko::kernels::reference::multivector::compute_sqrt(ref,
                                                       mtx->get_device_view());
    gko::kernels::GKO_DEVICE_NAMESPACE::multivector::compute_sqrt(
        exec, dmtx->get_device_view());

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<value_type>::value);
}
