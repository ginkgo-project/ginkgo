// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/eigensolver/lobpcg_kernels.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>

#include "core/test/utils.hpp"
#include "reference/base/lapack_bindings.hpp"


template <typename T>
class Lobpcg : public ::testing::Test {
protected:
    using value_type = T;
    using rc_value_type = gko::remove_complex<value_type>;
    using cmplx_value_type = gko::to_complex<rc_value_type>;
    using Mtx_r = gko::matrix::Dense<rc_value_type>;
    using Mtx_c = gko::to_complex<Mtx_r>;
    using Mtx = gko::matrix::Dense<value_type>;
    using Ary_r = gko::array<rc_value_type>;
    using Ary = gko::array<value_type>;
    Lobpcg() : exec(gko::ReferenceExecutor::create())
    {
        small_a_r = gko::initialize<Mtx_r>({{13.2, -4.3, -1.8, 0.12},
                                            {-4.3, 24.2, -1.7, -2.3},
                                            {-1.8, -1.7, 18.7, -0.8},
                                            {0.12, -2.3, -0.8, 10.0}},
                                           exec);
        small_b_r = gko::initialize<Mtx_r>({{2.0, -1.1, 0.3, 0.01},
                                            {-1.1, 2.5, -0.8, 0.5},
                                            {0.3, -0.8, 2.3, -0.2},
                                            {0.01, 0.5, -0.2, 1.9}},
                                           exec);
        small_a_cmplx = gko::initialize<Mtx_c>(
            {{cmplx_value_type{13.2, 0.0}, cmplx_value_type{-4.3, -0.3},
              cmplx_value_type{-1.8, 1.12}, cmplx_value_type{0.12, 0.6}},
             {cmplx_value_type{-4.3, 0.3}, cmplx_value_type{24.2, 0.0},
              cmplx_value_type{-1.7, -2.2}, cmplx_value_type{-2.3, -0.55}},
             {cmplx_value_type{-1.8, -1.12}, cmplx_value_type{-1.7, 2.2},
              cmplx_value_type{18.7, 0.0}, cmplx_value_type{-0.8, -1.18}},
             {cmplx_value_type{0.12, -0.6}, cmplx_value_type{-2.3, 0.55},
              cmplx_value_type{-0.8, 1.18}, cmplx_value_type{10.0, 0.0}}},
            exec);
        small_b_cmplx = gko::initialize<Mtx_c>(
            {{cmplx_value_type{2.0, 0.0}, cmplx_value_type{-1.1, -0.1},
              cmplx_value_type{0.3, 0.12}, cmplx_value_type{0.01, 0.4}},
             {cmplx_value_type{-1.1, 0.1}, cmplx_value_type{2.5, 0.0},
              cmplx_value_type{-0.8, -0.18}, cmplx_value_type{0.5, -0.097}},
             {cmplx_value_type{0.3, -0.12}, cmplx_value_type{-0.8, 0.18},
              cmplx_value_type{2.3, 0.0}, cmplx_value_type{-0.2, -0.172}},
             {cmplx_value_type{0.01, -0.4}, cmplx_value_type{0.5, 0.097},
              cmplx_value_type{-0.2, 0.172}, cmplx_value_type{1.9, 0.0}}},
            exec);
        small_e_vals = Ary_r(exec, 4);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx_r> small_a_r;
    std::shared_ptr<Mtx_r> small_b_r;
    std::shared_ptr<Mtx_c> small_a_cmplx;
    std::shared_ptr<Mtx_c> small_b_cmplx;
    Ary_r small_e_vals;
};

TYPED_TEST_SUITE(Lobpcg, gko::test::ValueTypesBase, TypenameNameGenerator);


TYPED_TEST(Lobpcg, KernelSymmEig)
{
    using Mtx_r = typename TestFixture::Mtx_r;
    using Mtx_c = typename TestFixture::Mtx_c;
    using Mtx = typename TestFixture::Mtx;
    using Ary_r = typename TestFixture::Ary_r;
    using Ary = typename TestFixture::Ary;
    using value_type = typename TestFixture::value_type;

    auto work = gko::array<char>(this->exec, 1);
    std::shared_ptr<Mtx> small_a;
    std::shared_ptr<Mtx> small_a_copy;

    if constexpr (gko::is_complex_s<value_type>::value) {
        small_a_copy = gko::clone(this->small_a_cmplx);
        // The kernel expects column-major, so transpose the matrices
        auto small_a_t =
            gko::share(gko::as<Mtx>(this->small_a_cmplx->transpose()));
        this->small_a_cmplx = small_a_t;

        small_a = this->small_a_cmplx;
    } else {
        small_a_copy = gko::clone(this->small_a_r);
        small_a = this->small_a_r;
    }

    gko::kernels::reference::lobpcg::symm_eig(this->exec, small_a.get(),
                                              &(this->small_e_vals), &work);

    // On exit, the eigenvectors will be stored in the A
    // matrix. We create submatrices for the vectors
    // to check that A * x = lambda * x for each vector.
    for (gko::size_type i = 0; i < this->small_e_vals.get_size(); i++) {
        auto evec = gko::share(Mtx::create(
            this->exec, gko::dim<2>{this->small_e_vals.get_size(), 1},
            Ary::view(
                this->exec, this->small_e_vals.get_size(),
                small_a->get_values() + i * this->small_e_vals.get_size()),
            1));

        auto lambda_r = gko::share(Mtx_r::create(
            this->exec, gko::dim<2>{1, 1},
            Ary_r::view(this->exec, 1, this->small_e_vals.get_data() + i), 1));
        std::shared_ptr<Mtx> lambda;
        if constexpr (gko::is_complex_s<value_type>::value) {
            lambda = lambda_r->make_complex();
        } else {
            lambda = lambda_r;
        }
        // A*x = lambda * x;
        auto a_x = Mtx::create(this->exec,
                               gko::dim<2>{this->small_e_vals.get_size(), 1});
        // a_x = A * x
        small_a_copy->apply(evec, a_x);
        // scale x by lambda
        evec->scale(lambda);

        GKO_ASSERT_MTX_NEAR(a_x, evec, r<value_type>::value);
    }
}


TYPED_TEST(Lobpcg, KernelSymmGeneralizedEig)
{
    using Mtx_r = typename TestFixture::Mtx_r;
    using Mtx_c = typename TestFixture::Mtx_c;
    using Mtx = typename TestFixture::Mtx;
    using Ary_r = typename TestFixture::Ary_r;
    using Ary = typename TestFixture::Ary;
    using value_type = typename TestFixture::value_type;

    auto work = gko::array<char>(this->exec, 1);
    std::shared_ptr<Mtx> small_a;
    std::shared_ptr<Mtx> small_b;

    std::shared_ptr<Mtx> small_a_copy;
    std::shared_ptr<Mtx> small_b_copy;

    if constexpr (gko::is_complex_s<value_type>::value) {
        small_a_copy = gko::clone(this->small_a_cmplx);
        small_b_copy = gko::clone(this->small_b_cmplx);
        // The kernel expects column-major, so transpose the matrices
        auto small_a_t =
            gko::share(gko::as<Mtx>(this->small_a_cmplx->transpose()));
        auto small_b_t =
            gko::share(gko::as<Mtx>(this->small_b_cmplx->transpose()));
        this->small_a_cmplx = small_a_t;
        this->small_b_cmplx = small_b_t;

        small_a = this->small_a_cmplx;
        small_b = this->small_b_cmplx;
    } else {
        small_a_copy = gko::clone(this->small_a_r);
        small_b_copy = gko::clone(this->small_b_r);
        small_a = this->small_a_r;
        small_b = this->small_b_r;
    }

    gko::kernels::reference::lobpcg::symm_generalized_eig(
        this->exec, small_a.get(), small_b.get(), &(this->small_e_vals), &work);

    // On exit, the eigenvectors will be stored in the A
    // matrix. We create submatrices for the vectors
    // to check that A * x = lambda * B * x for each vector.
    for (gko::size_type i = 0; i < this->small_e_vals.get_size(); i++) {
        auto evec = gko::share(Mtx::create(
            this->exec, gko::dim<2>{this->small_e_vals.get_size(), 1},
            Ary::view(
                this->exec, this->small_e_vals.get_size(),
                small_a->get_values() + i * this->small_e_vals.get_size()),
            1));

        auto lambda_r = gko::share(Mtx_r::create(
            this->exec, gko::dim<2>{1, 1},
            Ary_r::view(this->exec, 1, this->small_e_vals.get_data() + i), 1));
        std::shared_ptr<Mtx> lambda;
        if constexpr (gko::is_complex_s<value_type>::value) {
            lambda = lambda_r->make_complex();
        } else {
            lambda = lambda_r;
        }
        // A*x = lambda * B * x;
        auto a_x = Mtx::create(this->exec,
                               gko::dim<2>{this->small_e_vals.get_size(), 1});
        auto lambda_b_x = Mtx::create(
            this->exec, gko::dim<2>{this->small_e_vals.get_size(), 1});
        lambda_b_x->fill(gko::zero<value_type>());
        auto one = gko::initialize<Mtx>({gko::one<value_type>()}, this->exec);
        // a_x = A * x
        small_a_copy->apply(evec, a_x);
        // lambda_b_x = lambda * B * x
        small_b_copy->apply(lambda, evec, one, lambda_b_x);

        GKO_ASSERT_MTX_NEAR(a_x, lambda_b_x, r<value_type>::value);
    }
}


TYPED_TEST(Lobpcg, KernelBOrthonormalize)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using CsrMtx = gko::matrix::Csr<value_type, std::int32_t>;

    auto work = gko::array<char>(this->exec, 1);
    std::shared_ptr<Mtx> small_a;
    // Test with two kinds of B operator: Identity, and a Csr matrix
    auto id = gko::matrix::Identity<value_type>::create(
        this->exec, this->small_a_r->get_size()[0]);
    std::shared_ptr<CsrMtx> small_b_csr =
        gko::share(CsrMtx::create(this->exec, this->small_a_r->get_size()));
    // Create rectangular submatrix for testing
    if constexpr (gko::is_complex_s<value_type>::value) {
        small_a = this->small_a_cmplx->create_submatrix(
            gko::span{0, this->small_a_cmplx->get_size()[0]},
            gko::span{0, this->small_a_cmplx->get_size()[0] - 1});
        this->small_b_cmplx->convert_to(small_b_csr);
    } else {
        small_a = this->small_a_r->create_submatrix(
            gko::span{0, this->small_a_r->get_size()[0]},
            gko::span{0, this->small_a_cmplx->get_size()[0] - 1});
        this->small_b_r->convert_to(small_b_csr);
    }
    auto small_a_copy = gko::clone(small_a);

    // First, test with Identity operator as B
    gko::kernels::reference::lobpcg::b_orthonormalize(this->exec, small_a.get(),
                                                      id.get(), &work);
    // On exit, small_a should now be orthonormalized,
    // i.e., small_a^H * small_a = I.
    auto aH_a = Mtx::create(this->exec, gko::dim<2>{small_a->get_size()[1],
                                                    small_a->get_size()[1]});
    auto after_ortho_H = gko::as<Mtx>(small_a->conj_transpose());
    after_ortho_H->apply(small_a, aH_a);
    // Check if applying aH_a to the orthonormalized a^H leaves it unchanged
    auto result = Mtx::create(this->exec, after_ortho_H->get_size());
    aH_a->apply(after_ortho_H, result);
    GKO_ASSERT_MTX_NEAR(result, after_ortho_H, r<value_type>::value);

    // Now, test with Csr matrix operator as B
    gko::kernels::reference::lobpcg::b_orthonormalize(
        this->exec, small_a_copy.get(), small_b_csr.get(), &work);
    // On exit, small_a_copy should now be B-orthonormalized,
    // i.e., small_a_copy^H * small_b_csr * small_a_copy = I.
    auto b_a = Mtx::create(
        this->exec,
        gko::dim<2>{small_b_csr->get_size()[0], small_a_copy->get_size()[1]});
    small_b_csr->apply(small_a_copy, b_a);
    auto aH_b_a = Mtx::create(
        this->exec,
        gko::dim<2>{small_a_copy->get_size()[1], small_a_copy->get_size()[1]});
    auto after_b_ortho_H = gko::as<Mtx>(small_a_copy->conj_transpose());
    after_b_ortho_H->apply(b_a, aH_b_a);
    // Check if applying aH_b_a to the B-orthonormalized a^H leaves it unchanged
    result = Mtx::create(this->exec, after_b_ortho_H->get_size());
    aH_b_a->apply(after_b_ortho_H, result);
    GKO_ASSERT_MTX_NEAR(result, after_b_ortho_H, r<value_type>::value);
}
