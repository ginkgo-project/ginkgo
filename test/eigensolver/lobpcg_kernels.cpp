// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/eigensolver/lobpcg_kernels.hpp"

#include <algorithm>
#include <limits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


template <typename ValueType>
class Lobpcg : public CommonTestFixture {
protected:
    using rc_value_type = gko::remove_complex<value_type>;
    using cmplx_value_type = gko::to_complex<rc_value_type>;
    using Mtx_r = gko::matrix::Dense<rc_value_type>;
    using Mtx_c = gko::to_complex<Mtx_r>;
    using Mtx = gko::matrix::Dense<value_type>;
    using Ary_r = gko::array<rc_value_type>;
    using Ary = gko::array<value_type>;
    Lobpcg()
    {
        small_a_r = gko::initialize<Mtx_r>({{13.2, -4.3, -1.8, 0.12},
                                            {-4.3, 24.2, -1.7, -2.3},
                                            {-1.8, -1.7, 18.7, -0.8},
                                            {0.12, -2.3, -0.8, 10.0}},
                                           ref);
        small_b_r = gko::initialize<Mtx_r>({{2.0, -1.1, 0.3, 0.01},
                                            {-1.1, 2.5, -0.8, 0.5},
                                            {0.3, -0.8, 2.3, -0.2},
                                            {0.01, 0.5, -0.2, 1.9}},
                                           ref);
        small_a_cmplx = gko::initialize<Mtx_c>(
            {{cmplx_value_type{13.2, 0.0}, cmplx_value_type{-4.3, -0.3},
              cmplx_value_type{-1.8, 1.12}, cmplx_value_type{0.12, 0.6}},
             {cmplx_value_type{-4.3, 0.3}, cmplx_value_type{24.2, 0.0},
              cmplx_value_type{-1.7, -2.2}, cmplx_value_type{-2.3, -0.55}},
             {cmplx_value_type{-1.8, -1.12}, cmplx_value_type{-1.7, 2.2},
              cmplx_value_type{18.7, 0.0}, cmplx_value_type{-0.8, -1.18}},
             {cmplx_value_type{0.12, -0.6}, cmplx_value_type{-2.3, 0.55},
              cmplx_value_type{-0.8, 1.18}, cmplx_value_type{10.0, 0.0}}},
            ref);
        small_b_cmplx = gko::initialize<Mtx_c>(
            {{cmplx_value_type{2.0, 0.0}, cmplx_value_type{-1.1, -0.1},
              cmplx_value_type{0.3, 0.12}, cmplx_value_type{0.01, 0.4}},
             {cmplx_value_type{-1.1, 0.1}, cmplx_value_type{2.5, 0.0},
              cmplx_value_type{-0.8, -0.18}, cmplx_value_type{0.5, -0.097}},
             {cmplx_value_type{0.3, -0.12}, cmplx_value_type{-0.8, 0.18},
              cmplx_value_type{2.3, 0.0}, cmplx_value_type{-0.2, -0.172}},
             {cmplx_value_type{0.01, -0.4}, cmplx_value_type{0.5, 0.097},
              cmplx_value_type{-0.2, 0.172}, cmplx_value_type{1.9, 0.0}}},
            ref);
        small_e_vals = Ary_r(ref, 4);

        d_small_a_r = gko::clone(exec, small_a_r);
        d_small_b_r = gko::clone(exec, small_b_r);
        d_small_a_cmplx = gko::clone(exec, small_a_cmplx);
        d_small_b_cmplx = gko::clone(exec, small_b_cmplx);
        d_small_e_vals = Ary_r(exec, 4);
    }

    std::shared_ptr<Mtx_r> small_a_r;
    std::shared_ptr<Mtx_r> small_b_r;
    std::shared_ptr<Mtx_c> small_a_cmplx;
    std::shared_ptr<Mtx_c> small_b_cmplx;
    std::shared_ptr<Mtx> small_a;
    std::shared_ptr<Mtx> small_b;
    Ary_r small_e_vals;

    std::shared_ptr<Mtx_r> d_small_a_r;
    std::shared_ptr<Mtx_r> d_small_b_r;
    std::shared_ptr<Mtx_c> d_small_a_cmplx;
    std::shared_ptr<Mtx_c> d_small_b_cmplx;
    std::shared_ptr<Mtx> d_small_a;
    std::shared_ptr<Mtx> d_small_b;
    Ary_r d_small_e_vals;
};

TYPED_TEST_SUITE(Lobpcg, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Lobpcg, KernelSymmGeneralizedEigIsEquivalentToRef)
{
    using Mtx_r = typename TestFixture::Mtx_r;
    using Mtx = typename TestFixture::Mtx;
    using Ary_r = typename TestFixture::Ary_r;
    using Ary = typename TestFixture::Ary;
    using T = typename TestFixture::value_type;

    auto refwork = gko::array<char>(this->ref, 1);
    auto d_work = gko::array<char>(this->exec, 1);

    std::shared_ptr<Mtx> d_small_a_copy;
    std::shared_ptr<Mtx> d_small_b_copy;

    if constexpr (gko::is_complex_s<T>::value) {
        auto small_a_t =
            gko::share(gko::as<Mtx>(this->small_a_cmplx->transpose()));
        auto small_b_t =
            gko::share(gko::as<Mtx>(this->small_b_cmplx->transpose()));
        this->small_a_cmplx = small_a_t;
        this->small_b_cmplx = small_b_t;

        this->small_a = this->small_a_cmplx;
        this->small_b = this->small_b_cmplx;


        d_small_a_copy = gko::clone(this->d_small_a_cmplx);
        d_small_b_copy = gko::clone(this->d_small_b_cmplx);
        auto d_small_a_t =
            gko::share(gko::as<Mtx>(this->d_small_a_cmplx->transpose()));
        auto d_small_b_t =
            gko::share(gko::as<Mtx>(this->d_small_b_cmplx->transpose()));
        this->d_small_a_cmplx = d_small_a_t;
        this->d_small_b_cmplx = d_small_b_t;

        this->d_small_a = this->d_small_a_cmplx;
        this->d_small_b = this->d_small_b_cmplx;
    } else {
        d_small_a_copy = gko::clone(this->d_small_a_r);
        d_small_b_copy = gko::clone(this->d_small_b_r);
        this->small_a = this->small_a_r;
        this->small_b = this->small_b_r;
        this->d_small_a = this->d_small_a_r;
        this->d_small_b = this->d_small_b_r;
    }

    if (std::is_same_v<gko::remove_complex<T>, gko::half>) {
        EXPECT_THROW(gko::kernels::reference::lobpcg::symm_generalized_eig(
                         this->ref, this->small_a.get(), this->small_b.get(),
                         &(this->small_e_vals), &refwork),
                     gko::NotImplemented);
        EXPECT_THROW(
            gko::kernels::GKO_DEVICE_NAMESPACE::lobpcg::symm_generalized_eig(
                this->exec, this->d_small_a.get(), this->d_small_b.get(),
                &(this->d_small_e_vals), &d_work),
            gko::NotImplemented);
        return;
    } else {
        gko::kernels::reference::lobpcg::symm_generalized_eig(
            this->ref, this->small_a.get(), this->small_b.get(),
            &(this->small_e_vals), &refwork);
        gko::kernels::GKO_DEVICE_NAMESPACE::lobpcg::symm_generalized_eig(
            this->exec, this->d_small_a.get(), this->d_small_b.get(),
            &(this->d_small_e_vals), &d_work);

        const double tol =
            100 * std::numeric_limits<gko::remove_complex<T>>::epsilon();
        GKO_ASSERT_ARRAY_NEAR(this->d_small_e_vals, this->small_e_vals, tol);

        // The eigenvectors may differ by a factor of -1 between libraries.
        // Check for this and adjust before comparing output matrices.
        for (gko::size_type i = 0; i < this->small_e_vals.get_size(); i++) {
            auto evec = gko::share(Mtx::create(
                this->ref, gko::dim<2>{this->small_e_vals.get_size(), 1},
                Ary::view(this->ref, this->small_e_vals.get_size(),
                          this->small_a->get_values() +
                              i * this->small_e_vals.get_size()),
                1));
            T evec_first_entry = evec->at(0, 0);

            auto d_evec_start = gko::share(
                Mtx::create(this->exec, gko::dim<2>{1, 1},
                            Ary::view(this->exec, 1,
                                      this->d_small_a->get_values() +
                                          i * this->d_small_e_vals.get_size()),
                            1));
            T d_evec_first_entry;
            this->ref->copy_from(this->exec, 1, d_evec_start->get_values(),
                                 &d_evec_first_entry);

            auto neg_one = gko::initialize<Mtx>({-gko::one<T>()}, this->exec);
            if (gko::abs(evec_first_entry / d_evec_first_entry +
                         gko::one<T>()) < tol) {
                evec->scale(neg_one);
            }
        }
        GKO_ASSERT_MTX_NEAR(this->d_small_a, this->small_a, tol);
    }
}
