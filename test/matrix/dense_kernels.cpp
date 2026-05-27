// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include <ginkgo/core/matrix/multivector.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/components/fill_array_kernels.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class Dense : public CommonTestFixture {
protected:
    // in single mode, mixed_type will be the same as value_type
    using mixed_type = float;
    using Mtx = gko::matrix::Dense<value_type>;
    using ComplexMtx = gko::matrix::Dense<std::complex<value_type>>;
    using Diagonal = gko::matrix::Diagonal<value_type>;
    using Vec = gko::matrix::MultiVector<value_type>;
    using MixedVec = gko::matrix::MultiVector<mixed_type>;
    using ComplexVec = gko::matrix::MultiVector<std::complex<value_type>>;

    Dense() : rand_engine(15) {}

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
        y = gen_mtx<Vec>(1000, num_vecs);
        c_x = gen_mtx<ComplexMtx>(1000, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Vec>(1, num_vecs);
        } else {
            alpha = gko::initialize<Vec>({2.0}, ref);
        }
        dx = gko::clone(exec, x);
        dy = gko::clone(exec, y);
        dc_x = gko::clone(exec, c_x);
        dalpha = gko::clone(exec, alpha);
        result = Vec::create(ref, gko::dim<2>{1, num_vecs});
        dresult = Vec::create(exec, gko::dim<2>{1, num_vecs});
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(65, 25);
        y = gen_mtx<Vec>(25, 35);
        c_x = gen_mtx<ComplexMtx>(65, 25);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        result = gen_mtx<Vec>(65, 35);
        dx = gko::clone(exec, x);
        dy = gko::clone(exec, y);
        dc_x = gko::clone(exec, c_x);
        dresult = gko::clone(exec, result);
        dalpha = gko::clone(exec, alpha);
        dbeta = gko::clone(exec, beta);
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
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;
    std::unique_ptr<Vec> result;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<Vec> dresult;
};


TEST_F(Dense, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(y, result);
    dx->apply(dy, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, result, r<value_type>::value);
}


TEST_F(Dense, SimpleApplyMixedIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(convert<MixedVec>(y), convert<MixedVec>(result));
    dx->apply(convert<MixedVec>(dy), convert<MixedVec>(dresult));

    GKO_ASSERT_MTX_NEAR(dresult, result, 1e-7);
}


TEST_F(Dense, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(alpha, y, beta, result);
    dx->apply(dalpha, dy, dbeta, dresult);

    GKO_ASSERT_MTX_NEAR(dresult, result, r<value_type>::value);
}


TEST_F(Dense, AdvancedApplyMixedIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(convert<MixedVec>(alpha), convert<MixedVec>(y),
             convert<MixedVec>(beta), convert<MixedVec>(result));
    dx->apply(convert<MixedVec>(dalpha), convert<MixedVec>(dy),
              convert<MixedVec>(dbeta), convert<MixedVec>(dresult));

    GKO_ASSERT_MTX_NEAR(dresult, result, 1e-7);
}


TEST_F(Dense, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(x->get_size()[1], 1);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(x->get_size()[0], 1);
    auto dcomplex_x = gko::clone(exec, complex_x);

    x->apply(complex_b, complex_x);
    dx->apply(dcomplex_b, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<value_type>::value);
}


TEST_F(Dense, ApplyToMixedComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(x->get_size()[1], 1);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(x->get_size()[0], 1);
    auto dcomplex_x = gko::clone(exec, complex_x);

    x->apply(complex_b, complex_x);
    dx->apply(dcomplex_b, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 2e-7);
}


TEST_F(Dense, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(x->get_size()[1], 1);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(x->get_size()[0], 1);
    auto dcomplex_x = gko::clone(exec, complex_x);

    x->apply(alpha, complex_b, beta, complex_x);
    dx->apply(dalpha, dcomplex_b, dbeta, dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, r<value_type>::value);
}


TEST_F(Dense, AdvancedApplyToMixedComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexVec>(x->get_size()[1], 1);
    auto dcomplex_b = gko::clone(exec, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(x->get_size()[0], 1);
    auto dcomplex_x = gko::clone(exec, complex_x);

    x->apply(convert<MixedVec>(alpha), complex_b, convert<MixedVec>(beta),
             complex_x);
    dx->apply(convert<MixedVec>(dalpha), dcomplex_b, convert<MixedVec>(dbeta),
              dcomplex_x);

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 2e-7);
}


TEST_F(Dense, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx*>(dtrans.get()),
                        static_cast<Mtx*>(trans.get()), 0);
}


TEST_F(Dense, IsTransposableIntoDenseCrossExecutor)
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


TEST_F(Dense, IsConjugateTransposable)
{
    set_up_apply_data();

    auto trans = c_x->conj_transpose();
    auto dtrans = dc_x->conj_transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx*>(dtrans.get()),
                        static_cast<ComplexMtx*>(trans.get()), 0);
}


TEST_F(Dense, IsConjugateTransposableIntoDenseCrossExecutor)
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


TEST_F(Dense, ExtractDiagonalOnTallSkinnyIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = x->extract_diagonal();
    auto ddiag = dx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Dense, ExtractDiagonalOnTallSkinnyIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto diag = Diagonal::create(ref, x->get_size()[1]);
    // test make_temporary_clone
    auto ddiag = Diagonal::create(ref, x->get_size()[1]);

    x->extract_diagonal(diag);
    dx->extract_diagonal(ddiag);

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Dense, ExtractDiagonalOnShortFatIsEquivalentToRef)
{
    set_up_apply_data();

    auto diag = y->extract_diagonal();
    auto ddiag = dy->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Dense, ExtractDiagonalOnShortFatIntoDenseCrossExecutor)
{
    set_up_apply_data();
    auto diag = Diagonal::create(ref, y->get_size()[0]);
    // test make_temporary_clone
    auto ddiag = Diagonal::create(ref, y->get_size()[0]);

    y->extract_diagonal(diag);
    dy->extract_diagonal(ddiag);

    GKO_ASSERT_MTX_NEAR(diag, ddiag, 0);
}


TEST_F(Dense, CopyRespectsStride)
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


TEST_F(Dense, FillIsEquivalentToRef)
{
    set_up_vector_data(3);

    x->fill(42);
    dx->fill(42);

    GKO_ASSERT_MTX_NEAR(dx, x, 0);
}


TEST_F(Dense, StridedFillIsEquivalentToRef)
{
    using T = value_type;
    auto x = gko::initialize<gko::matrix::Dense<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, ref);
    auto dx = gko::initialize<gko::matrix::Dense<T>>(
        4, {I<T>{1.0, 2.0}, I<T>{3.0, 4.0}, I<T>{5.0, 6.0}}, exec);

    x->fill(42);
    dx->fill(42);

    GKO_ASSERT_MTX_NEAR(dx, x, 0);
}
