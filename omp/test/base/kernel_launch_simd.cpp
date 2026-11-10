// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "omp/base/kernel_launch_simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <omp.h>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "common/unified/base/kernel_launch_solver.hpp"
#include "core/test/utils.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace cg {


/*
 * Not decorative: real bodies live in gko::kernels::<backend>::<algo> and call
 * zero/one/abs/conj/squared_norm unqualified, so the function below must be
 * compiled from exactly this scope to prove the simd:: overloads win partial
 * ordering against the gko:: ones.
 */
namespace simd = gko::kernels::omp::simd;
namespace hn = hwy::HWY_NAMESPACE;


template <typename Access>
void unqualified_helper_body(const double* xp, double* yp, std::size_t count)
{
    using D = hn::ScalableTag<double>;
    simd::ref_leaf<D, Access> x{xp, count};
    simd::ref<D, Access> y{yp, count};

    // All five helpers and every operator, unqualified.
    y = zero(y) + one(y) * abs(x) + conj(x) - squared_norm(x) / one(y);
    y = -y;
    y += x;
    y -= zero(y);
    y *= one(y);
    y /= one(y);
}


// Both access forms, so neither the full nor the tail instantiation is skipped.
template void unqualified_helper_body<simd::full_access>(const double*, double*,
                                                         std::size_t);
template void unqualified_helper_body<simd::partial_access>(const double*,
                                                            double*,
                                                            std::size_t);


}  // namespace cg
}  // namespace omp
}  // namespace kernels
}  // namespace gko


namespace {


namespace simd = gko::kernels::omp::simd;
namespace hn = hwy::HWY_NAMESPACE;

using D = hn::ScalableTag<double>;


// Applies `body` over `n` contiguous elements using the expression layer,
// mirroring what the Phase 2 launcher will do: full vectors, then one partial
// iteration for the tail.
template <typename Body>
void run_over(std::size_t n, Body body)
{
    const D d;
    const auto lanes = hn::Lanes(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        body(simd::full_access{}, i, lanes);
    }
    if (i < n) {
        body(simd::partial_access{}, i, n - i);
    }
}


class KernelLaunchSimd : public ::testing::Test {
protected:
    // 1003 is not a multiple of any lane count, so the tail is always
    // exercised.
    static constexpr std::size_t n = 1003;

    KernelLaunchSimd() : x(n), y(n), y_ref(n)
    {
        for (std::size_t i = 0; i < n; i++) {
            x[i] = 0.5 * static_cast<double>(i) + 1.0;
            y[i] = y_ref[i] = 1.0 / (static_cast<double>(i) + 1.0);
        }
    }

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> y_ref;
};


TEST_F(KernelLaunchSimd, AxpyIsBitIdenticalToScalar)
{
    // std::fma, not `+= alpha * x`: the layer emits MulAdd for this shape, and
    // so does the scalar path wherever the compiler contracts (the default).
    const double alpha = 2.5;
    for (std::size_t i = 0; i < n; i++) {
        y_ref[i] = std::fma(alpha, x[i], y_ref[i]);
    }

    run_over(n, [&](auto access, std::size_t i, std::size_t count) {
        using access_t = decltype(access);
        simd::ref_leaf<D, access_t> xv{x.data() + i, count};
        simd::ref<D, access_t> yv{y.data() + i, count};
        yv += alpha * xv;
    });

    for (std::size_t i = 0; i < n; i++) {
        ASSERT_EQ(y_ref[i], y[i]) << "at index " << i;
    }
}


TEST_F(KernelLaunchSimd, TwoLevelExpressionIsBitIdenticalToScalar)
{
    // Shape of cg::step_1: p = z + tmp * p
    std::vector<double> p(n), z(n), p_ref(n);
    for (std::size_t i = 0; i < n; i++) {
        p[i] = p_ref[i] = 0.25 * static_cast<double>(i) - 3.0;
        z[i] = 2.0 - 0.125 * static_cast<double>(i);
    }
    // Fused for the same reason as AxpyIsBitIdenticalToScalar above.
    const double tmp = 1.75;
    for (std::size_t i = 0; i < n; i++) {
        p_ref[i] = std::fma(tmp, p_ref[i], z[i]);
    }

    run_over(n, [&](auto access, std::size_t i, std::size_t count) {
        using access_t = decltype(access);
        simd::ref_leaf<D, access_t> zv{z.data() + i, count};
        simd::ref<D, access_t> pv{p.data() + i, count};
        pv = zv + tmp * pv;
    });

    for (std::size_t i = 0; i < n; i++) {
        ASSERT_EQ(p_ref[i], p[i]) << "at index " << i;
    }
}


TEST_F(KernelLaunchSimd, RefAssignmentStoresRatherThanCopyingThePointer)
{
    // Guards the hand-written copy-assignment operator: the implicit one would
    // copy `ptr` and leave memory untouched.
    std::vector<double> dst(n, 0.0);

    run_over(n, [&](auto access, std::size_t i, std::size_t count) {
        using access_t = decltype(access);
        simd::ref<D, access_t> src{x.data() + i, count};
        simd::ref<D, access_t> out{dst.data() + i, count};
        out = src;
    });

    for (std::size_t i = 0; i < n; i++) {
        ASSERT_EQ(x[i], dst[i]) << "at index " << i;
    }
}


TEST_F(KernelLaunchSimd, PartialTailDoesNotWriteBeyondCount)
{
    // StoreN must not touch lanes past `count`.
    std::vector<double> buf(n + 16, -1.0);
    const D d;
    const auto lanes = hn::Lanes(d);
    // Derive the partial count from the actual lane count. A hardcoded value
    // silently fails on narrow vectors: SVE2 on NVIDIA Grace is 128-bit, so
    // `double` has only 2 lanes and any count >= 2 is not a partial vector.
    if (lanes < 2) {
        GTEST_SKIP() << "needs at least 2 lanes to form a partial vector";
    }
    const std::size_t count = lanes - 1;

    simd::ref<D, simd::partial_access> out{buf.data(), count};
    out = simd::scalar_leaf<D, double>{7.0};

    for (std::size_t i = 0; i < count; i++) {
        ASSERT_EQ(7.0, buf[i]) << "at index " << i;
    }
    for (std::size_t i = count; i < buf.size(); i++) {
        ASSERT_EQ(-1.0, buf[i]) << "wrote past count at index " << i;
    }
}


TEST_F(KernelLaunchSimd, SquaredNormEvaluatesOperandOnce)
{
    std::vector<double> out(n, 0.0);
    run_over(n, [&](auto access, std::size_t i, std::size_t count) {
        using access_t = decltype(access);
        simd::ref_leaf<D, access_t> xv{x.data() + i, count};
        simd::ref<D, access_t> ov{out.data() + i, count};
        ov = simd::squared_norm(xv);
    });
    for (std::size_t i = 0; i < n; i++) {
        ASSERT_EQ(x[i] * x[i], out[i]) << "at index " << i;
    }
}


TEST_F(KernelLaunchSimd, UnqualifiedHelpersResolveToTheSimdOverloads)
{
    // The regression guard for the ambiguity between gko::zero/one/abs/conj/
    // squared_norm and their simd:: counterparts. The call site is
    // gko::kernels::omp::cg::unqualified_helper_body above; that it compiles
    // at all is the point, and this test also checks it computes what the
    // unqualified names mean.
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; i++) {
        out[i] = 0.0;
    }

    run_over(n, [&](auto access, std::size_t i, std::size_t count) {
        using access_t = decltype(access);
        gko::kernels::omp::cg::unqualified_helper_body<access_t>(
            x.data() + i, out.data() + i, count);
    });

    for (std::size_t i = 0; i < n; i++) {
        const double xi = x[i];
        double expected =
            0.0 + 1.0 * std::abs(xi) + xi - (xi * xi) / 1.0;  // helpers
        expected = -expected;                                 // unary minus
        expected += xi;
        expected -= 0.0;
        expected *= 1.0;
        expected /= 1.0;
        ASSERT_EQ(expected, out[i]) << "at index " << i;
    }
}


template <typename T, typename = void>
struct has_eq_int : std::false_type {};

template <typename T>
struct has_eq_int<T, std::void_t<decltype(std::declval<const T&>() == 0)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_add_int : std::false_type {};

template <typename T>
struct has_add_int<T, std::void_t<decltype(std::declval<const T&>() + 0)>>
    : std::true_type {};


TEST_F(KernelLaunchSimd, IndexHasNoComparisonOrArithmeticOperators)
{
    // The compile-time guard that keeps non-vectorizable bodies off the SIMD
    // path: `if (row == 0)` and `row + 1` must be ill-formed against
    // simd::index, not merely unconvertible.
    static_assert(!has_eq_int<simd::index>::value,
                  "`simd::index == 0` must not compile");
    static_assert(!has_add_int<simd::index>::value,
                  "`simd::index + 0` must not compile");
    // A positive control, so the detection idiom above is known to detect.
    static_assert(has_eq_int<int>::value, "detection idiom is broken");
    static_assert(has_add_int<int>::value, "detection idiom is broken");
}


// Mirror what run_kernel_simd does internally -- map the arguments exactly as
// it does, then ask the dispatcher whether the resulting pack is eligible --
// so a test can assert that the SIMD path was actually taken. Without this,
// every correctness test below passes byte-identically when the launcher
// silently falls back.
template <typename KernelFunction, typename... KernelArgs>
bool simd_takes_simd_path(std::shared_ptr<const gko::OmpExecutor> exec,
                          KernelFunction fn, gko::dim<2> size,
                          KernelArgs&&... args)
{
    return gko::kernels::omp::try_run_kernel_simd_impl(
        exec, fn, size, gko::kernels::omp::map_to_device(args)...);
}


/** @see simd_takes_simd_path */
template <typename KernelFunction, typename... KernelArgs>
bool solver_simd_takes_simd_path(std::shared_ptr<const gko::OmpExecutor> exec,
                                 KernelFunction fn, gko::dim<2> size,
                                 gko::size_type default_stride,
                                 KernelArgs&&... args)
{
    return gko::kernels::omp::try_run_kernel_simd_impl(
        exec, fn, size,
        gko::kernels::omp::map_to_device_solver(
            args, static_cast<gko::int64>(default_stride))...);
}


/**
 * @see simd_takes_simd_path
 *
 * The reduction equivalent. Without it ReductionTailIsMaskedToIdentityNotZero
 * would prove nothing: a scalar max-reduction over the same all-negative input
 * also comes out to -1, with no masking involved.
 */
template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
bool reduction_simd_takes_simd_path(
    std::shared_ptr<const gko::OmpExecutor> exec, KernelFunction fn,
    ReductionOp op, FinalizeOp finalize, ValueType identity, ValueType* result,
    gko::dim<2> size, gko::array<char>& tmp, KernelArgs&&... args)
{
    return gko::kernels::omp::try_run_kernel_reduction_simd_impl(
        exec, fn, op, finalize, identity, result, size, tmp,
        gko::kernels::omp::map_to_device(args)...);
}


/**
 * An op advertising the vector interface but implementing half of it. A gate
 * probing only reduce_lanes would admit it and then fail inside the launcher
 * instead of declining to the scalar path.
 */
struct reduce_lanes_only_op {
    using gko_simd_reduction = void;

    template <typename U, typename V>
    auto operator()(U a, V b) const
    {
        return a + b;
    }

    template <typename D>
    static hn::TFromD<D> reduce_lanes(D d, hn::Vec<D> v)
    {
        return hn::ReduceSum(d, v);
    }
};

static_assert(!gko::kernels::omp::has_simd_reduction<reduce_lanes_only_op>,
              "an op with reduce_lanes but no vector_op must decline");
static_assert(gko::kernels::omp::has_simd_reduction<
                  gko::kernels::omp::reduce_sum_op<double>>,
              "the shipped sum op must be eligible");
static_assert(gko::kernels::omp::has_simd_reduction<
                  gko::kernels::omp::reduce_max_op<float>>,
              "the ScalableTag<double> probe must answer for a float op too, "
              "because the functors' ValueType parameter is unused");


template <typename ValueType>
class KernelLaunchSimdLauncher : public ::testing::Test {
protected:
    using value_type = ValueType;
    using Dense = gko::matrix::Dense<ValueType>;

    KernelLaunchSimdLauncher() : exec(gko::OmpExecutor::create()) {}

    static gko::size_type lanes()
    {
        return static_cast<gko::size_type>(
            hn::Lanes(hn::ScalableTag<ValueType>()));
    }

    // Every size that distinguishes a loop tier from its neighbour. On a
    // 1-lane target `lanes - 1` is 0, which is not a size a Dense can have, so
    // it only joins the list when it is positive.
    static std::vector<gko::size_type> interesting_sizes()
    {
        const auto l = lanes();
        std::vector<gko::size_type> sizes{1, l, l + 1, 4 * l, 4 * l + 1, 1003};
        if (l > 1) {
            sizes.push_back(l - 1);
        }
        return sizes;
    }

    std::shared_ptr<const gko::OmpExecutor> exec;
};


// float and double have different lane counts and different
// simd_chunk_alignment, so both have to be compiled and run.
using SimdValueTypes = ::testing::Types<double, float>;

TYPED_TEST_SUITE(KernelLaunchSimdLauncher, SimdValueTypes,
                 TypenameNameGenerator);


// The reduction tests below only exercise `double`: the tail size and the
// tolerance both derive from its lane count and precision. `TEST_F` needs a
// concrete fixture, so this aliases the `double` instantiation above.
using KernelLaunchSimdLauncherDouble = KernelLaunchSimdLauncher<double>;


TYPED_TEST(KernelLaunchSimdLauncher, AddScaledMatchesScalarLauncherExactly)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto exec = this->exec;
    for (auto n : TestFixture::interesting_sizes()) {
        auto x = Dense::create(exec, gko::dim<2>{n, 1});
        auto y_simd = Dense::create(exec, gko::dim<2>{n, 1});
        auto y_probe = Dense::create(exec, gko::dim<2>{n, 1});
        auto y_scalar = Dense::create(exec, gko::dim<2>{n, 1});
        for (gko::size_type i = 0; i < n; i++) {
            x->at(i, 0) =
                static_cast<value_type>(0.5 * static_cast<double>(i) + 1.0);
            const auto y0 =
                static_cast<value_type>(1.0 / (static_cast<double>(i) + 1.0));
            y_simd->at(i, 0) = y_probe->at(i, 0) = y_scalar->at(i, 0) = y0;
        }
        const value_type alpha{2.5};

        auto body = [] GKO_KERNEL(auto row, auto col, auto alpha, auto x,
                                  auto y) {
            y(row, col) += alpha * x(row, col);
        };
        gko::kernels::omp::run_kernel(exec, body, gko::dim<2>{n, 1}, alpha,
                                      x->get_const_device_view(),
                                      y_scalar->get_device_view());
        gko::kernels::omp::run_kernel_simd(exec, body, gko::dim<2>{n, 1}, alpha,
                                           x->get_const_device_view(),
                                           y_simd->get_device_view());
        // Same body, same argument shape, on its own output: proves this pack
        // reaches the SIMD path rather than passing the comparison below by
        // falling back.
        ASSERT_TRUE(simd_takes_simd_path(exec, body, gko::dim<2>{n, 1}, alpha,
                                         x->get_const_device_view(),
                                         y_probe->get_device_view()))
            << "n = " << n;

        // This TU is built with -ffp-contract=off, so the scalar launcher does
        // not fuse while the SIMD path always does; the library, which
        // contracts by default, produces the fused result on both paths.
        for (gko::size_type i = 0; i < n; i++) {
            const auto y0 =
                static_cast<value_type>(1.0 / (static_cast<double>(i) + 1.0));
            const auto fused = std::fma(alpha, x->at(i, 0), y0);
            ASSERT_EQ(fused, y_simd->at(i, 0))
                << "n = " << n << ", index " << i;
            ASSERT_EQ(fused, y_probe->at(i, 0))
                << "n = " << n << ", index " << i;
            ASSERT_EQ(static_cast<value_type>(y0 + alpha * x->at(i, 0)),
                      y_scalar->at(i, 0))
                << "n = " << n << ", index " << i;
        }
    }
}


TYPED_TEST(KernelLaunchSimdLauncher, FallsBackToScalarForMultipleColumns)
{
    // cols > 1 is not eligible; the launcher must still produce correct results
    // via the scalar path.
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto exec = this->exec;
    const gko::size_type n = 37;
    auto x = Dense::create(exec, gko::dim<2>{n, 3});
    auto y = Dense::create(exec, gko::dim<2>{n, 3});
    for (gko::size_type i = 0; i < n; i++) {
        for (gko::size_type j = 0; j < 3; j++) {
            x->at(i, j) = static_cast<value_type>(i * 3 + j);
            y->at(i, j) = value_type{};
        }
    }

    auto body = [] GKO_KERNEL(auto row, auto col, auto x, auto y) {
        y(row, col) = x(row, col) + x(row, col);
    };
    ASSERT_FALSE(simd_takes_simd_path(exec, body, gko::dim<2>{n, 3},
                                      x->get_const_device_view(),
                                      y->get_device_view()));
    gko::kernels::omp::run_kernel_simd(exec, body, gko::dim<2>{n, 3},
                                       x->get_const_device_view(),
                                       y->get_device_view());

    for (gko::size_type i = 0; i < n; i++) {
        for (gko::size_type j = 0; j < 3; j++) {
            ASSERT_EQ(static_cast<value_type>(i * 3 + j) * value_type{2},
                      y->at(i, j));
        }
    }
}


TYPED_TEST(KernelLaunchSimdLauncher, FallsBackToScalarForNonUnitStride)
{
    // A column view of a wider Dense has stride > 1 and is not eligible.
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto exec = this->exec;
    const gko::size_type n = 41;
    auto wide = Dense::create(exec, gko::dim<2>{n, 2});
    for (gko::size_type i = 0; i < n; i++) {
        wide->at(i, 0) = static_cast<value_type>(i);
        wide->at(i, 1) = value_type{-1};
    }
    auto col = wide->create_submatrix(gko::span{0, n}, gko::span{0, 1});
    ASSERT_GT(col->get_stride(), 1u) << "test requires a strided view";

    gko::kernels::omp::run_kernel_simd(
        exec,
        [] GKO_KERNEL(auto row, auto c, auto factor, auto v) {
            v(row, c) *= factor;
        },
        gko::dim<2>{n, 1}, value_type{3}, col->get_device_view());

    for (gko::size_type i = 0; i < n; i++) {
        ASSERT_EQ(static_cast<value_type>(i) * value_type{3}, wide->at(i, 0));
        ASSERT_EQ(value_type{-1}, wide->at(i, 1))
            << "clobbered the neighbouring column";
    }
}


TYPED_TEST(KernelLaunchSimdLauncher, EligibilityPredicateAgreesWithExpectation)
{
    // Without this, a silently ineligible kernel would pass every correctness
    // test above by taking the scalar path.
    using Dense = typename TestFixture::Dense;
    using value_type = typename TestFixture::value_type;
    auto exec = this->exec;
    if (TestFixture::lanes() <= 1) {
        GTEST_SKIP() << "build resolved to a 1-lane Highway target; "
                        "configure with -DGINKGO_OMP_SIMD_ARCH=native";
    }
    auto vec = Dense::create(exec, gko::dim<2>{16, 1});
    // map_to_device is a forwarding reference: to hit the matrix::view::dense
    // specialization (rather than the identity-passthrough primary template)
    // it must be called on a named lvalue, exactly as run_kernel_simd itself
    // does internally, not directly on the get_device_view() prvalue.
    auto view = vec->get_device_view();
    auto acc = gko::kernels::omp::map_to_device(view);
    ASSERT_TRUE(gko::kernels::omp::simd_detail::all_unit_stride(acc));

    auto wide = Dense::create(exec, gko::dim<2>{16, 2});
    auto col = wide->create_submatrix(gko::span{0, 16}, gko::span{0, 1});
    auto strided_view = col->get_device_view();
    auto strided_acc = gko::kernels::omp::map_to_device(strided_view);
    ASSERT_FALSE(gko::kernels::omp::simd_detail::all_unit_stride(strided_acc));
}


TYPED_TEST(KernelLaunchSimdLauncher, SolverLauncherMatchesScalarLauncherExactly)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto exec = this->exec;
    for (auto n : TestFixture::interesting_sizes()) {
        auto alpha = Dense::create(exec, gko::dim<2>{1, 1});
        alpha->at(0, 0) = value_type{2.5};
        auto x = Dense::create(exec, gko::dim<2>{n, 1});
        auto y_simd = Dense::create(exec, gko::dim<2>{n, 1});
        auto y_probe = Dense::create(exec, gko::dim<2>{n, 1});
        auto y_scalar = Dense::create(exec, gko::dim<2>{n, 1});
        for (gko::size_type i = 0; i < n; i++) {
            x->at(i, 0) =
                static_cast<value_type>(0.5 * static_cast<double>(i) + 1.0);
            const auto y0 =
                static_cast<value_type>(1.0 / (static_cast<double>(i) + 1.0));
            y_simd->at(i, 0) = y_probe->at(i, 0) = y_scalar->at(i, 0) = y0;
        }

        // The shape of a real solver kernel: a row_vector scalar operand plus
        // default_stride-wrapped dense operands, whose stride comes from the
        // launch rather than from the accessors themselves.
        auto body = [] GKO_KERNEL(auto row, auto col, auto alpha, auto x,
                                  auto y) {
            y(row, col) += alpha[col] * x(row, col);
        };
        gko::kernels::omp::run_kernel_solver(
            exec, body, gko::dim<2>{n, 1}, 1,
            gko::kernels::omp::row_vector(alpha->get_const_device_view()),
            gko::kernels::omp::default_stride(x->get_const_device_view()),
            gko::kernels::omp::default_stride(y_scalar->get_device_view()));
        gko::kernels::omp::run_kernel_solver_simd(
            exec, body, gko::dim<2>{n, 1}, 1,
            gko::kernels::omp::row_vector(alpha->get_const_device_view()),
            gko::kernels::omp::default_stride(x->get_const_device_view()),
            gko::kernels::omp::default_stride(y_simd->get_device_view()));
        ASSERT_TRUE(solver_simd_takes_simd_path(
            exec, body, gko::dim<2>{n, 1}, 1,
            gko::kernels::omp::row_vector(alpha->get_const_device_view()),
            gko::kernels::omp::default_stride(x->get_const_device_view()),
            gko::kernels::omp::default_stride(y_probe->get_device_view())))
            << "n = " << n;

        // @see AddScaledMatchesScalarLauncherExactly for why the two paths are
        // compared against different references in this TU.
        for (gko::size_type i = 0; i < n; i++) {
            const auto y0 =
                static_cast<value_type>(1.0 / (static_cast<double>(i) + 1.0));
            const auto a = alpha->at(0, 0);
            const auto fused = std::fma(a, x->at(i, 0), y0);
            ASSERT_EQ(fused, y_simd->at(i, 0))
                << "n = " << n << ", index " << i;
            ASSERT_EQ(fused, y_probe->at(i, 0))
                << "n = " << n << ", index " << i;
            ASSERT_EQ(static_cast<value_type>(y0 + a * x->at(i, 0)),
                      y_scalar->at(i, 0))
                << "n = " << n << ", index " << i;
        }
    }
}


TYPED_TEST(KernelLaunchSimdLauncher, SolverLauncherFallsBackForNonUnitStride)
{
    // Eligibility of a default_stride operand is decided by the stride handed
    // to the launch, not by anything visible on the operand itself.
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Dense;
    auto exec = this->exec;
    const gko::size_type n = 41;
    auto alpha = Dense::create(exec, gko::dim<2>{1, 1});
    alpha->at(0, 0) = value_type{3};
    auto wide = Dense::create(exec, gko::dim<2>{n, 2});
    for (gko::size_type i = 0; i < n; i++) {
        wide->at(i, 0) = static_cast<value_type>(i);
        wide->at(i, 1) = value_type{-1};
    }
    ASSERT_EQ(wide->get_stride(), 2u);

    auto body = [] GKO_KERNEL(auto row, auto col, auto alpha, auto v) {
        v(row, col) *= alpha[col];
    };
    ASSERT_FALSE(solver_simd_takes_simd_path(
        exec, body, gko::dim<2>{n, 1}, wide->get_stride(),
        gko::kernels::omp::row_vector(alpha->get_const_device_view()),
        gko::kernels::omp::default_stride(wide->get_device_view())));
    gko::kernels::omp::run_kernel_solver_simd(
        exec, body, gko::dim<2>{n, 1}, wide->get_stride(),
        gko::kernels::omp::row_vector(alpha->get_const_device_view()),
        gko::kernels::omp::default_stride(wide->get_device_view()));

    for (gko::size_type i = 0; i < n; i++) {
        ASSERT_EQ(static_cast<value_type>(i) * value_type{3}, wide->at(i, 0));
        ASSERT_EQ(value_type{-1}, wide->at(i, 1))
            << "clobbered the neighbouring column";
    }
}


class KernelLaunchSimdEligibility : public ::testing::Test {
protected:
    KernelLaunchSimdEligibility() : exec(gko::OmpExecutor::create()) {}

    std::shared_ptr<const gko::OmpExecutor> exec;
};


TEST_F(KernelLaunchSimdEligibility, MixedScalarAndValueTypeDoesNotVectorize)
{
    // Correctness, not scope: scalar `float x *= double alpha` promotes,
    // computes in double and rounds once, while the SIMD path would broadcast
    // hn::Set(d, static_cast<float>(alpha)) and round alpha first. Ginkgo
    // passes alpha as a raw const ScalarType*/scalar, never as an accessor, so
    // the guard has to look at non-accessor operands to see this at all.
    const gko::size_type n = 64;
    auto x = gko::matrix::Dense<float>::create(exec, gko::dim<2>{n, 1});
    x->fill(1.0f);

    auto body = [] GKO_KERNEL(auto row, auto col, auto alpha, auto v) {
        v(row, col) *= alpha;
    };

    ASSERT_FALSE(simd_takes_simd_path(exec, body, gko::dim<2>{n, 1}, double{3},
                                      x->get_device_view()));
    ASSERT_TRUE(simd_takes_simd_path(exec, body, gko::dim<2>{n, 1}, float{3},
                                     x->get_device_view()));
    for (gko::size_type i = 0; i < n; i++) {
        ASSERT_EQ(3.0f, x->at(i, 0)) << "index " << i;
    }
}


TEST_F(KernelLaunchSimdEligibility, NonFloatingPointOperandsStayNeutral)
{
    // stopping_status arrays and index pointers are not floating point, so
    // they must contribute nothing to the shared value type and must not
    // exclude an otherwise eligible launch.
    const gko::size_type n = 64;
    auto v = gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, 1});
    v->fill(1.0);
    gko::array<gko::stopping_status> stop(exec, 1);
    stop.get_data()[0].reset();
    gko::array<gko::int32> idx(exec, 1);
    idx.get_data()[0] = 0;

    ASSERT_TRUE(simd_takes_simd_path(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto v, auto stop, auto idx) {
            if (!stop[col].has_stopped() && idx[col] == 0) {
                v(row, col) += v(row, col);
            }
        },
        gko::dim<2>{n, 1}, v->get_device_view(), stop, idx));

    for (gko::size_type i = 0; i < n; i++) {
        ASSERT_EQ(2.0, v->at(i, 0)) << "index " << i;
    }
}


TEST_F(KernelLaunchSimdEligibility, FillShapeStaysEligible)
{
    // dense::fill: one accessor plus a scalar of the same type.
    const gko::size_type n = 64;
    auto v = gko::matrix::Dense<double>::create(exec, gko::dim<2>{n, 1});
    v->fill(0.0);

    ASSERT_TRUE(simd_takes_simd_path(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto value, auto v) {
            v(row, col) = value;
        },
        gko::dim<2>{n, 1}, 1.5, v->get_device_view()));

    for (gko::size_type i = 0; i < n; i++) {
        ASSERT_EQ(1.5, v->at(i, 0)) << "index " << i;
    }
}


TEST_F(KernelLaunchSimdEligibility, ScalarOnlyLaunchIsNotEligible)
{
    // A scalar contributes a value type, but with no accessor there is nothing
    // to vectorize over and the body would run rows/lanes times instead of
    // rows times.
    ASSERT_FALSE(simd_takes_simd_path(
        exec, [] GKO_KERNEL(auto row, auto col, auto value) {},
        gko::dim<2>{64, 1}, 1.0));
}


TEST(KernelLaunchSimdPartition, AlignedThreadRangeIsAnExactCover)
{
    // The 4-argument overload takes the team size explicitly, so it needs no
    // parallel region. This decides which rows each thread owns.
    using gko::kernels::omp::simd_detail::aligned_thread_range;
    for (gko::int64 rows = 0; rows <= 64; rows++) {
        for (gko::int64 align : {1, 2, 8, 16}) {
            for (gko::int64 threads = 1; threads <= 8; threads++) {
                gko::int64 cursor = 0;
                for (gko::int64 id = 0; id < threads; id++) {
                    SCOPED_TRACE(::testing::Message()
                                 << "rows = " << rows << ", align = " << align
                                 << ", threads = " << threads
                                 << ", thread_id = " << id);
                    const auto range =
                        aligned_thread_range(rows, align, threads, id);
                    // Contiguous with the previous range: no gap, no overlap.
                    ASSERT_EQ(cursor, range.first);
                    ASSERT_LE(range.first, range.second);
                    ASSERT_LE(range.second, rows);
                    // Only the end of [0, rows) may be unaligned; a thread
                    // past the last block starts there and owns nothing.
                    if (range.first != rows) {
                        ASSERT_EQ(0, range.first % align);
                    }
                    if (range.second != rows) {
                        ASSERT_EQ(0, range.second % align);
                    }
                    cursor = range.second;
                }
                ASSERT_EQ(rows, cursor)
                    << "rows = " << rows << ", align = " << align
                    << ", threads = " << threads;
            }
        }
    }
}


TEST_F(KernelLaunchSimdLauncherDouble, DotMatchesScalarLauncherWithinTolerance)
{
    // Not bit-identical by design: lane count and four accumulators change the
    // summation order. The OMP reduction is already thread-count dependent, so
    // this widens existing non-determinism rather than introducing a new kind.
    for (auto n : interesting_sizes()) {
        auto x = Dense::create(exec, gko::dim<2>{n, 1});
        auto y = Dense::create(exec, gko::dim<2>{n, 1});
        for (gko::size_type i = 0; i < n; i++) {
            x->at(i, 0) = 0.5 * static_cast<double>(i) + 1.0;
            y->at(i, 0) = 1.0 / (static_cast<double>(i) + 1.0);
        }
        double simd_result = 0.0;
        double scalar_result = 0.0;
        gko::array<char> tmp{exec};

        auto body = [] GKO_KERNEL(auto i, auto j, auto x, auto y) {
            return x(i, j) * y(i, j);
        };
        gko::kernels::omp::run_kernel_col_reduction_cached(
            exec, body, GKO_KERNEL_REDUCE_SUM(double), &scalar_result,
            gko::dim<2>{n, 1}, tmp, x->get_const_device_view(),
            y->get_const_device_view());
        gko::kernels::omp::run_kernel_col_reduction_cached_simd(
            exec, body, GKO_KERNEL_REDUCE_SUM(double), &simd_result,
            gko::dim<2>{n, 1}, tmp, x->get_const_device_view(),
            y->get_const_device_view());
        // Same body, same argument shape, on its own output: proves this pack
        // reaches the SIMD path rather than passing the comparison below by
        // falling back to the scalar launcher under a different name.
        double probe_result = 0.0;
        ASSERT_TRUE(reduction_simd_takes_simd_path(
            exec, body, GKO_KERNEL_REDUCE_SUM(double), &probe_result,
            gko::dim<2>{n, 1}, tmp, x->get_const_device_view(),
            y->get_const_device_view()))
            << "n = " << n;

        ASSERT_NEAR(scalar_result, simd_result,
                    std::abs(scalar_result) * 1e-13 + 1e-13)
            << "n = " << n;
    }
}


TEST_F(KernelLaunchSimdLauncherDouble, BenchmarkDotShapeReachesTheSimdPath)
{
    // benchmark/blas builds `dot` from two default-stride Dense at cols == 1
    // and measures no gain at any size, while `norm` on the same launcher
    // gains. Pinning the shape down makes that a performance question rather
    // than an undetected fallback to the scalar launcher.
    const gko::size_type n = 1 << 16;
    auto x = Dense::create(exec, gko::dim<2>{n, 1});
    auto y = Dense::create(exec, gko::dim<2>{n, 1});
    x->fill(1.0);
    y->fill(1.0);
    ASSERT_EQ(1u, x->get_stride());
    ASSERT_EQ(1u, y->get_stride());
    double result = 0.0;
    gko::array<char> tmp{exec};

    ASSERT_TRUE(reduction_simd_takes_simd_path(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto x, auto y) {
            return x(i, j) * y(i, j);
        },
        GKO_KERNEL_REDUCE_SUM(double), &result, gko::dim<2>{n, 1}, tmp,
        x->get_const_device_view(), y->get_const_device_view()));
    ASSERT_EQ(static_cast<double>(n), result);
}


TEST_F(KernelLaunchSimdLauncherDouble, ReductionTailIsMaskedToIdentityNotZero)
{
    // A max-reduction over all-negative input: if the tail were zero-filled
    // instead of masked to identity, the result would wrongly come out as 0.
    const auto lanes =
        static_cast<gko::size_type>(hn::Lanes(hn::ScalableTag<double>()));
    if (lanes < 2) {
        // With one lane per vector the 1x loop consumes every element and the
        // tail is empty, so the masking this test exists for never runs and
        // the assertion below would pass vacuously.
        GTEST_SKIP() << "needs at least 2 lanes to form a partial vector";
    }
    const auto n = 4 * lanes + 3;  // guarantees a non-empty tail
    auto x = Dense::create(exec, gko::dim<2>{n, 1});
    for (gko::size_type i = 0; i < n; i++) {
        x->at(i, 0) = -1.0 - static_cast<double>(i);
    }
    double result = 1.0;
    gko::array<char> tmp{exec};

    auto body = [] GKO_KERNEL(auto i, auto j, auto x) { return x(i, j); };
    // Not GKO_KERNEL_REDUCE_MAX(double): that macro bakes in identity =
    // double{} (0.0), which is wrong for max and would shift every argument
    // after it by one position. The op and finalize it would have supplied
    // are spelled out here instead, so the true max identity (-infinity) can
    // be passed explicitly.
    auto op = gko::kernels::omp::reduce_max_op<double>{};
    auto finalize = [] GKO_KERNEL(auto a) { return a; };
    const double identity = -std::numeric_limits<double>::infinity();
    // A plain scalar max-reduction over this same all-negative input also
    // comes out to -1, with no masking involved at all: this assertion is
    // what makes the ASSERT_EQ below actually test the masking logic, rather
    // than passing vacuously because the launcher silently fell back.
    double probe_result = 1.0;
    ASSERT_TRUE(reduction_simd_takes_simd_path(
        exec, body, op, finalize, identity, &probe_result, gko::dim<2>{n, 1},
        tmp, x->get_const_device_view()));

    gko::kernels::omp::run_kernel_col_reduction_cached_simd(
        exec, body, op, finalize, identity, &result, gko::dim<2>{n, 1}, tmp,
        x->get_const_device_view());

    ASSERT_EQ(-1.0, result);
}


TEST_F(KernelLaunchSimdLauncherDouble, ReductionWithRawLambdaOpDeclinesAtImpl)
{
    // run_kernel_col_reduction_cached_simd now static_asserts on a raw lambda
    // op, so only the lower-level probe can be exercised with one here.
    const gko::size_type n = 101;
    auto x = Dense::create(exec, gko::dim<2>{n, 1});
    double result = 0.0;
    gko::array<char> tmp{exec};

    auto square_body = [] GKO_KERNEL(auto i, auto j, auto x) {
        return x(i, j) * x(i, j);
    };
    auto raw_sum = [] GKO_KERNEL(auto a, auto b) { return a + b; };
    auto raw_finalize = [] GKO_KERNEL(auto a) { return a; };
    // A raw lambda has no vector_op/reduce_lanes, so has_simd_reduction is
    // false and try_run_kernel_reduction_simd_impl must decline.
    ASSERT_FALSE(reduction_simd_takes_simd_path(
        exec, square_body, raw_sum, raw_finalize, double{}, &result,
        gko::dim<2>{n, 1}, tmp, x->get_const_device_view()));
}


TEST_F(KernelLaunchSimdLauncherDouble, ReductionDeclinesForMultipleColumns)
{
    // Once Phase 4 converts the call sites this becomes the *common* path: a
    // column reduction over a multi-vector has one result per column, which
    // the single-accumulator SIMD loop cannot produce, so it must decline and
    // hand over to the scalar launcher unchanged.
    const gko::size_type n = 37;
    const gko::size_type cols = 3;
    auto x = Dense::create(exec, gko::dim<2>{n, cols});
    for (gko::size_type i = 0; i < n; i++) {
        for (gko::size_type j = 0; j < cols; j++) {
            x->at(i, j) = 0.5 * static_cast<double>(i) + static_cast<double>(j);
        }
    }
    std::vector<double> simd_result(cols, 0.0);
    std::vector<double> scalar_result(cols, 0.0);
    gko::array<char> tmp{exec};

    auto body = [] GKO_KERNEL(auto i, auto j, auto x) { return x(i, j); };
    ASSERT_FALSE(reduction_simd_takes_simd_path(
        exec, body, GKO_KERNEL_REDUCE_SUM(double), simd_result.data(),
        gko::dim<2>{n, cols}, tmp, x->get_const_device_view()));

    gko::kernels::omp::run_kernel_col_reduction_cached(
        exec, body, GKO_KERNEL_REDUCE_SUM(double), scalar_result.data(),
        gko::dim<2>{n, cols}, tmp, x->get_const_device_view());
    gko::kernels::omp::run_kernel_col_reduction_cached_simd(
        exec, body, GKO_KERNEL_REDUCE_SUM(double), simd_result.data(),
        gko::dim<2>{n, cols}, tmp, x->get_const_device_view());

    // Bit-identical, not merely close: the declined launch must reach exactly
    // the same scalar code the plain launcher does.
    for (gko::size_type j = 0; j < cols; j++) {
        ASSERT_EQ(scalar_result[j], simd_result[j]) << "column " << j;
    }
}


TEST_F(KernelLaunchSimdLauncherDouble, ReductionDeclinesForNonUnitStride)
{
    // One column of a wider Dense has stride 2, so consecutive rows are not
    // contiguous and there is nothing for a contiguous load to load.
    const gko::size_type n = 37;
    auto wide = Dense::create(exec, gko::dim<2>{n, 2});
    for (gko::size_type i = 0; i < n; i++) {
        wide->at(i, 0) = 0.5 * static_cast<double>(i) + 1.0;
        wide->at(i, 1) = -1.0;
    }
    auto col = wide->create_submatrix(gko::span{0, n}, gko::span{0, 1});
    double simd_result = 0.0;
    double scalar_result = 0.0;
    gko::array<char> tmp{exec};

    auto body = [] GKO_KERNEL(auto i, auto j, auto x) { return x(i, j); };
    ASSERT_FALSE(reduction_simd_takes_simd_path(
        exec, body, GKO_KERNEL_REDUCE_SUM(double), &simd_result,
        gko::dim<2>{n, 1}, tmp, col->get_const_device_view()));

    gko::kernels::omp::run_kernel_col_reduction_cached(
        exec, body, GKO_KERNEL_REDUCE_SUM(double), &scalar_result,
        gko::dim<2>{n, 1}, tmp, col->get_const_device_view());
    gko::kernels::omp::run_kernel_col_reduction_cached_simd(
        exec, body, GKO_KERNEL_REDUCE_SUM(double), &simd_result,
        gko::dim<2>{n, 1}, tmp, col->get_const_device_view());

    ASSERT_EQ(scalar_result, simd_result);
    // The neighbouring column must not have been folded in.
    ASSERT_EQ(0.5 * 36.0 * 37.0 / 2.0 + 37.0, simd_result);
}


TEST_F(KernelLaunchSimdLauncherDouble,
       ReductionIsCorrectWhenTeamIsSmallerThanRequested)
{
    // omp_get_max_threads() is the *requested* team size; the runtime may
    // grant fewer, leaving slots of `partial` unwritten but still summed.
    // `tmp` is pre-sized so the launcher cannot reallocate it, and filled with
    // 0xFF bytes, which read back as NaN, so an unwritten slot poisons the
    // result outright rather than shifting it plausibly.
    const auto max_threads = omp_get_max_threads();
    if (max_threads < 2) {
        GTEST_SKIP() << "needs at least 2 threads for the team to shrink";
    }
    // Confirm this runtime actually serialises the nested region; if it
    // honours the request there is no smaller team to exercise.
    int inner_team = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
#pragma omp parallel num_threads(2)
            {
                if (omp_get_thread_num() == 0) {
                    inner_team = omp_get_num_threads();
                }
            }
        }
    }
    if (inner_team >= max_threads) {
        GTEST_SKIP() << "this runtime grants nested teams the full size ("
                     << inner_team << "); nothing to shrink";
    }

    const gko::size_type n = 1000;
    auto x = Dense::create(exec, gko::dim<2>{n, 1});
    double expected = 0.0;
    for (gko::size_type i = 0; i < n; i++) {
        x->at(i, 0) = 0.5 * static_cast<double>(i) + 1.0;
        expected += x->at(i, 0);
    }
    gko::array<char> tmp{
        exec, sizeof(double) * static_cast<gko::size_type>(max_threads + 4)};
    std::fill_n(tmp.get_data(), tmp.get_size(), static_cast<char>(0xFF));
    double result = 0.0;

#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            gko::kernels::omp::run_kernel_col_reduction_cached_simd(
                exec, [] GKO_KERNEL(auto i, auto j, auto x) { return x(i, j); },
                GKO_KERNEL_REDUCE_SUM(double), &result, gko::dim<2>{n, 1}, tmp,
                x->get_const_device_view());
        }
    }

    ASSERT_FALSE(std::isnan(result)) << "summed an unwritten partial slot";
    ASSERT_NEAR(expected, result, expected * 1e-13);
}


}  // namespace
