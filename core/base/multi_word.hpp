// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_MULTI_WORD_HPP_
#define GKO_CORE_BASE_MULTI_WORD_HPP_


#include <cmath>
#include <limits>
#include <type_traits>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
/**
 * Compile-time options and algorithms of multi_word.
 *
 * References:
 * - Joldes, Muller, Popescu, "Tight and rigorous error bounds for basic
 *   building blocks of double-word arithmetic", 2017,
 *   https://doi.org/10.1145/3121432 (double-word arithmetic, the algorithm
 *   numbers below refer to this paper)
 * - Muller, Rideau, "Formalization of double-word arithmetic, and comments on
 *   'Tight and rigorous error bounds for basic building blocks of double-word
 *   arithmetic'", 2022 (corrected bound of DWTimesDW2)
 * - Lange, Rump, "Faithfully rounded floating-point computations", 2020,
 *   https://doi.org/10.1145/3290955 (pair arithmetic)
 * - Noack, "twofloat", https://github.com/esa-tu-darmstadt/twofloat, which
 *   implements both. multi_word uses the same algorithms for N = 2.
 */
namespace mw {


/**
 * When results are normalized.
 */
enum class normalization {
    /**
     * After every operation (double-word arithmetic, Joldes et al. 2017).
     */
    eager,
    /**
     * Only when normalize() is called (pair arithmetic, Lange and Rump 2020):
     * cheaper, but the error bound grows with the number of dependent
     * operations. The algorithm option has no effect.
     */
    none
};


/**
 * Which algorithm variant the operations use, see multi_word for the
 * algorithms and their error bounds.
 */
enum class algorithm {
    /**
     * Fewer operations. The relative error of addition is unbounded when the
     * leading words cancel.
     */
    fast,
    /**
     * Proven error bounds for all inputs.
     */
    accurate
};


/**
 * How infinities and NaN are treated.
 */
enum class special {
    /**
     * No extra checks. The error-free transformations turn infinities into
     * NaN, e.g. inf + 1 is NaN, and an overflow gives NaN instead of inf.
     */
    ignore,
    /**
     * A result with an infinite or NaN word is replaced by the operation on
     * the leading words with zero lower words, so that inf + 1 is inf and an
     * overflow gives inf, as for a single word. The check costs a comparison
     * per operation (per word with normalization::none, which can leave NaN
     * in a lower word only), the leading words' operation is only computed if
     * it fails.
     *
     * If an intermediate step of an algorithm overflows or gives NaN although
     * the exact result is finite, the result falls back to the operation on
     * the leading words, so to single-word accuracy. This happens e.g. in the
     * splitting of TwoProd without FMA for words above about max / 2^(p/2),
     * and in DWDivDW3 for divisors below 1 / max or x / inf.
     */
    handle
};


/**
 * The compile-time options of a multi_word.
 *
 * @tparam Normalization  when results are normalized
 * @tparam Algorithm  the algorithm variant of the operations
 * @tparam Fma  whether the operations may use fused multiply-add. Without it,
 *              the exact product of two words needs Dekker's TwoProd, and the
 *              accurate product and quotient of two multi_words are not
 *              available.
 * @tparam Special  how infinities and NaN are treated
 */
template <normalization Normalization = normalization::eager,
          algorithm Algorithm = algorithm::accurate, bool Fma = true,
          special Special = special::ignore>
struct policy {
    static constexpr normalization norm = Normalization;
    static constexpr algorithm alg = Algorithm;
    static constexpr bool fma = Fma;
    static constexpr special special_values = Special;
};


namespace detail {


/**
 * A fixed number of words, the most significant one first.
 */
template <typename T, int K>
struct word_array {
    GKO_ATTRIBUTES constexpr T& operator[](int i) noexcept { return data[i]; }

    GKO_ATTRIBUTES constexpr const T& operator[](int i) const noexcept
    {
        return data[i];
    }

    T data[K];
};


/*
 * The basic operations, rounded to nearest. On the device, the intrinsics
 * keep the compiler from contracting a multiplication and an addition into an
 * FMA, which would make the results differ from the host. On the host, this
 * relies on the compiler not contracting either, which is the default unless
 * the target has FMA instructions (e.g. -march=native). The error-free
 * transformations stay exact under contraction, the other operations only get
 * more accurate.
 */


template <typename T>
GKO_ATTRIBUTES GKO_INLINE T add_rn(T a, T b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>) {
        return __fadd_rn(a, b);
    } else {
        return __dadd_rn(a, b);
    }
#else
    return a + b;
#endif
}


template <typename T>
GKO_ATTRIBUTES GKO_INLINE T sub_rn(T a, T b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>) {
        return __fsub_rn(a, b);
    } else {
        return __dsub_rn(a, b);
    }
#else
    return a - b;
#endif
}


template <typename T>
GKO_ATTRIBUTES GKO_INLINE T mul_rn(T a, T b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>) {
        return __fmul_rn(a, b);
    } else {
        return __dmul_rn(a, b);
    }
#else
    return a * b;
#endif
}


template <typename T>
GKO_ATTRIBUTES GKO_INLINE T div_rn(T a, T b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>) {
        return __fdiv_rn(a, b);
    } else {
        return __ddiv_rn(a, b);
    }
#else
    return a / b;
#endif
}


/**
 * Returns a * b + c with a single rounding.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE T fma_rn(T a, T b, T c)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>) {
        return __fmaf_rn(a, b, c);
    } else {
        return __fma_rn(a, b, c);
    }
#else
    return std::fma(a, b, c);
#endif
}


/*
 * Error-free transformations. Each returns the rounded result and its
 * rounding error, which sum up to the exact result if nothing overflows or
 * underflows.
 */


/**
 * TwoSum (Knuth, Algorithm 2) for any a and b.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> two_sum(T a, T b)
{
    const T s = add_rn(a, b);
    const T a_rounded = sub_rn(s, b);
    const T b_rounded = sub_rn(s, a_rounded);
    const T a_error = sub_rn(a, a_rounded);
    const T b_error = sub_rn(b, b_rounded);
    return {{s, add_rn(a_error, b_error)}};
}


/**
 * Fast2Sum (Dekker, Algorithm 1), if a == 0 or the exponent of a is at least
 * the exponent of b.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> fast_two_sum(T a, T b)
{
    const T s = add_rn(a, b);
    const T b_rounded = sub_rn(s, a);
    return {{s, sub_rn(b, b_rounded)}};
}


/**
 * Veltkamp's splitting: returns two words that sum up to a, each with at most
 * half of the significand bits, so that their products are exact. Overflows
 * for |a| above about max / 2^(p/2).
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> split(T a)
{
    constexpr int s = (std::numeric_limits<T>::digits + 1) / 2;
    constexpr T factor = static_cast<T>((1 << s) + 1);
    const T scaled = mul_rn(factor, a);
    const T upper = add_rn(scaled, sub_rn(a, scaled));
    return {{upper, sub_rn(a, upper)}};
}


/**
 * TwoProd: with FMA, the error is a single FMA (Fast2Mult, Algorithm 3).
 * Without FMA, it is Dekker's algorithm, which multiplies the halves from
 * Veltkamp's splitting exactly.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> two_prod(T a, T b)
{
    const T product = mul_rn(a, b);
    if constexpr (Fma) {
        return {{product, fma_rn(a, b, -product)}};
    } else {
        const auto a_split = split(a);
        const auto b_split = split(b);
        T error = sub_rn(mul_rn(a_split[0], b_split[0]), product);
        error = add_rn(error, mul_rn(a_split[0], b_split[1]));
        error = add_rn(error, mul_rn(a_split[1], b_split[0]));
        return {{product, add_rn(error, mul_rn(a_split[1], b_split[1]))}};
    }
}


/**
 * Returns a - q * b exactly, for q = fl(a / b). The remainder of a correctly
 * rounded division is a floating-point number.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE T division_residual(T a, T q, T b)
{
    if constexpr (Fma) {
        return fma_rn(-q, b, a);
    } else {
        // q * b is within a factor of two of a, so a - p[0] is exact
        const auto p = two_prod<false>(q, b);
        return sub_rn(sub_rn(a, p[0]), p[1]);
    }
}


/*
 * Double-word algorithms (Joldes et al. 2017). x and y are normalized double
 * words, a is a single word, u = 2^-p is the unit roundoff of T. The bounds
 * are relative errors.
 */


/**
 * Algorithm 4 (DWPlusFP), x + a, 2u^2.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_plus_fp(const word_array<T, 2>& x,
                                                      T a)
{
    const auto s = two_sum(x[0], a);
    return fast_two_sum(s[0], add_rn(x[1], s[1]));
}


/**
 * Algorithm 5 (SloppyDWPlusDW), x + y, 3u^2 if x and y have the same sign,
 * otherwise unbounded.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> sloppy_dw_plus_dw(
    const word_array<T, 2>& x, const word_array<T, 2>& y)
{
    const auto s = two_sum(x[0], y[0]);
    return fast_two_sum(s[0], add_rn(s[1], add_rn(x[1], y[1])));
}


/**
 * Algorithm 6 (AccurateDWPlusDW), x + y, 3u^2 / (1 - 4u).
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> accurate_dw_plus_dw(
    const word_array<T, 2>& x, const word_array<T, 2>& y)
{
    const auto s = two_sum(x[0], y[0]);
    const auto t = two_sum(x[1], y[1]);
    const auto v = fast_two_sum(s[0], add_rn(s[1], t[0]));
    return fast_two_sum(v[0], add_rn(t[1], v[1]));
}


/**
 * Algorithm 7 (DWTimesFP1), x * a, 1.5u^2 + 4u^3.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_times_fp1(
    const word_array<T, 2>& x, T a)
{
    const auto c = two_prod<Fma>(x[0], a);
    const auto t = fast_two_sum(c[0], mul_rn(x[1], a));
    return fast_two_sum(t[0], add_rn(t[1], c[1]));
}


/**
 * Algorithm 8 (DWTimesFP2), x * a, 3u^2.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_times_fp2(
    const word_array<T, 2>& x, T a)
{
    const auto c = two_prod<Fma>(x[0], a);
    return fast_two_sum(c[0], add_rn(c[1], mul_rn(x[1], a)));
}


/**
 * Algorithm 9 (DWTimesFP3), x * a with FMA, 2u^2.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_times_fp3(
    const word_array<T, 2>& x, T a)
{
    const auto c = two_prod<true>(x[0], a);
    return fast_two_sum(c[0], fma_rn(x[1], a, c[1]));
}


/**
 * Algorithm 10 (DWTimesDW1), x * y, 7u^2.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_times_dw1(
    const word_array<T, 2>& x, const word_array<T, 2>& y)
{
    const auto c = two_prod<Fma>(x[0], y[0]);
    const T cross = add_rn(mul_rn(x[0], y[1]), mul_rn(x[1], y[0]));
    return fast_two_sum(c[0], add_rn(c[1], cross));
}


/**
 * Algorithm 11 (DWTimesDW2), x * y with FMA, 5u^2 / (1 + u)^2 (Muller and
 * Rideau 2022, the 6u^2 of Joldes et al. had a flawed proof).
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_times_dw2(
    const word_array<T, 2>& x, const word_array<T, 2>& y)
{
    const auto c = two_prod<true>(x[0], y[0]);
    const T cross = fma_rn(x[1], y[0], mul_rn(x[0], y[1]));
    return fast_two_sum(c[0], add_rn(c[1], cross));
}


/**
 * Algorithm 12 (DWTimesDW3), x * y with FMA, 5u^2.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_times_dw3(
    const word_array<T, 2>& x, const word_array<T, 2>& y)
{
    const auto c = two_prod<true>(x[0], y[0]);
    const T lower = fma_rn(x[0], y[1], mul_rn(x[1], y[1]));
    const T cross = fma_rn(x[1], y[0], lower);
    return fast_two_sum(c[0], add_rn(c[1], cross));
}


/**
 * Algorithm 15 (DWDivFP3), x / a, 3u^2.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_div_fp3(const word_array<T, 2>& x,
                                                      T a)
{
    const T q = div_rn(x[0], a);
    const auto p = two_prod<Fma>(q, a);
    // exact
    const T residual = sub_rn(x[0], p[0]);
    const T delta = add_rn(sub_rn(residual, p[1]), x[1]);
    return fast_two_sum(q, div_rn(delta, a));
}


/**
 * Algorithm 17 (DWDivDW2), x / y, 15u^2 + 56u^3. As in twofloat, y * q uses
 * DWTimesFP3 with FMA and DWTimesFP1 (as in the paper) without.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_div_dw2(const word_array<T, 2>& x,
                                                      const word_array<T, 2>& y)
{
    const T q = div_rn(x[0], y[0]);
    word_array<T, 2> r{};
    if constexpr (Fma) {
        r = dw_times_fp3(y, q);
    } else {
        r = dw_times_fp1<false>(y, q);
    }
    // exact
    const T residual = sub_rn(x[0], r[0]);
    const T delta = add_rn(residual, sub_rn(x[1], r[1]));
    return fast_two_sum(q, div_rn(delta, y[0]));
}


/**
 * Algorithm 18 (DWDivDW3), x / y with FMA, 9.8u^2. It multiplies x by the
 * reciprocal of y with DWTimesDW3 as in the paper; twofloat uses DWTimesDW2.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_div_dw3(const word_array<T, 2>& x,
                                                      const word_array<T, 2>& y)
{
    const T r = div_rn(T{1}, y[0]);
    // exact
    const T residual = fma_rn(-y[0], r, T{1});
    const auto e = fast_two_sum(residual, -mul_rn(y[1], r));
    const auto delta = dw_times_fp3(e, r);
    const auto reciprocal = dw_plus_fp(delta, r);
    return dw_times_dw3(x, reciprocal);
}


/*
 * Pair arithmetic (Lange and Rump 2020): the leading word is the result of
 * the operation on the leading words, the second word approximates its error.
 * The results are not normalized, so the inputs need not be either.
 */


/**
 * CPairSum, x + y.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> pair_sum(const word_array<T, 2>& x,
                                                    const word_array<T, 2>& y)
{
    const auto s = two_sum(x[0], y[0]);
    return {{s[0], add_rn(s[1], add_rn(x[1], y[1]))}};
}


/**
 * x + a, derived from CPairSum.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> pair_sum_fp(
    const word_array<T, 2>& x, T a)
{
    const auto s = two_sum(x[0], a);
    return {{s[0], add_rn(s[1], x[1])}};
}


/**
 * CPairProd, x * y.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> pair_prod(const word_array<T, 2>& x,
                                                     const word_array<T, 2>& y)
{
    const auto c = two_prod<Fma>(x[0], y[0]);
    const T cross = add_rn(mul_rn(x[0], y[1]), mul_rn(x[1], y[0]));
    return {{c[0], add_rn(c[1], cross)}};
}


/**
 * x * a, derived from CPairProd.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> pair_prod_fp(
    const word_array<T, 2>& x, T a)
{
    const auto c = two_prod<Fma>(x[0], a);
    return {{c[0], add_rn(c[1], mul_rn(x[1], a))}};
}


/**
 * CPairDiv, x / y.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> pair_div(const word_array<T, 2>& x,
                                                    const word_array<T, 2>& y)
{
    const T q = div_rn(x[0], y[0]);
    const T numerator = sub_rn(
        add_rn(division_residual<Fma>(x[0], q, y[0]), x[1]), mul_rn(q, y[1]));
    return {{q, div_rn(numerator, add_rn(y[0], y[1]))}};
}


/**
 * x / a, derived from CPairDiv.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> pair_div_fp(
    const word_array<T, 2>& x, T a)
{
    const T q = div_rn(x[0], a);
    const T numerator = add_rn(division_residual<Fma>(x[0], q, a), x[1]);
    return {{q, div_rn(numerator, a)}};
}


template <typename T>
GKO_ATTRIBUTES GKO_INLINE bool is_finite(T value)
{
    constexpr auto inf = std::numeric_limits<T>::infinity();
    return value > -inf && value < inf;
}


/**
 * Implements special::handle for any number of words.
 *
 * @tparam Normalized  whether the result is normalized. Then a lower word can
 *                     only be infinite or NaN if the leading word is, so only
 *                     the leading word is checked.
 *
 * @param result  the result of the multi-word operation
 * @param leading  computes the same operation on the leading words only, which
 *                 is what a single word would give. It is only called if the
 *                 result is not finite.
 */
template <bool Normalized, typename T, int N, typename Leading>
GKO_ATTRIBUTES GKO_INLINE word_array<T, N> handle_special_values(
    const word_array<T, N>& result, Leading leading)
{
    bool finite = is_finite(result[0]);
    if constexpr (!Normalized) {
#pragma unroll
        for (int i = 1; i < N; ++i) {
            finite = finite && is_finite(result[i]);
        }
    }
    if (finite) {
        return result;
    }
    // The operation on the leading words differs from the exact result by a
    // few ulps at most, so the exact result can only overflow if this one is
    // close to max. Otherwise, an operand is not finite, or an intermediate
    // step overflowed or gave NaN, e.g. fma(-inf, 0, 1) in DWDivDW3 for
    // x / inf, and the result on the leading words is the best one left.
    constexpr auto inf = std::numeric_limits<T>::infinity();
    constexpr auto near_overflow =
        std::numeric_limits<T>::max() *
        (T{1} - 4 * std::numeric_limits<T>::epsilon());
    const T value = leading();
    word_array<T, N> fixed{};
    if (value >= near_overflow) {
        fixed[0] = inf;
    } else if (value <= -near_overflow) {
        fixed[0] = -inf;
    } else {
        fixed[0] = value;
    }
    return fixed;
}


/**
 * Selects the algorithms for N words and a policy. Each N needs its own
 * algorithms, so each N specializes this class.
 */
template <typename T, int N, typename Policy>
struct arithmetic {
    static_assert(N == 2,
                  "multi_word arithmetic is only implemented for N = 2");
};


template <typename T, typename Policy>
struct arithmetic<T, 2, Policy> {
    using words = word_array<T, 2>;

    static constexpr bool pair = Policy::norm == normalization::none;
    static constexpr bool fast = Policy::alg == algorithm::fast;
    static constexpr bool fma = Policy::fma;

    /**
     * Normalizes words whose leading word has the largest exponent, e.g. the
     * words of a value rounded word by word. For them, Fast2Sum is exact.
     */
    static GKO_ATTRIBUTES GKO_INLINE words normalize_ordered(const words& x)
    {
        return fast_two_sum(x[0], x[1]);
    }

    /**
     * Normalizes the words of a result. The double-word algorithms keep the
     * leading word the largest, so Fast2Sum suffices. Pair arithmetic can
     * leave a second word with a larger exponent, which needs TwoSum.
     */
    static GKO_ATTRIBUTES GKO_INLINE words normalize(const words& x)
    {
        if constexpr (pair) {
            return two_sum(x[0], x[1]);
        } else {
            return normalize_ordered(x);
        }
    }

    static GKO_ATTRIBUTES GKO_INLINE words add(const words& x, const words& y)
    {
        if constexpr (pair) {
            return pair_sum(x, y);
        } else if constexpr (fast) {
            return sloppy_dw_plus_dw(x, y);
        } else {
            return accurate_dw_plus_dw(x, y);
        }
    }

    static GKO_ATTRIBUTES GKO_INLINE words add(const words& x, T a)
    {
        if constexpr (pair) {
            return pair_sum_fp(x, a);
        } else {
            return dw_plus_fp(x, a);
        }
    }

    static GKO_ATTRIBUTES GKO_INLINE words mul(const words& x, const words& y)
    {
        static_assert(pair || fma || fast,
                      "the accurate product of two multi_words needs FMA "
                      "(DWTimesDW3), use algorithm::fast without FMA");
        if constexpr (pair) {
            return pair_prod<fma>(x, y);
        } else if constexpr (fma && fast) {
            return dw_times_dw2(x, y);
        } else if constexpr (fma) {
            return dw_times_dw3(x, y);
        } else {
            return dw_times_dw1<false>(x, y);
        }
    }

    static GKO_ATTRIBUTES GKO_INLINE words mul(const words& x, T a)
    {
        if constexpr (pair) {
            return pair_prod_fp<fma>(x, a);
        } else if constexpr (fma) {
            return dw_times_fp3(x, a);
        } else if constexpr (fast) {
            return dw_times_fp2<false>(x, a);
        } else {
            return dw_times_fp1<false>(x, a);
        }
    }

    static GKO_ATTRIBUTES GKO_INLINE words div(const words& x, const words& y)
    {
        static_assert(pair || fma || fast,
                      "the accurate quotient of two multi_words needs FMA "
                      "(DWDivDW3), use algorithm::fast without FMA");
        if constexpr (pair) {
            return pair_div<fma>(x, y);
        } else if constexpr (fast) {
            return dw_div_dw2<fma>(x, y);
        } else {
            return dw_div_dw3(x, y);
        }
    }

    static GKO_ATTRIBUTES GKO_INLINE words div(const words& x, T a)
    {
        if constexpr (pair) {
            return pair_div_fp<fma>(x, a);
        } else {
            return dw_div_fp3<fma>(x, a);
        }
    }
};


}  // namespace detail
}  // namespace mw


/**
 * A floating-point number represented as the unevaluated sum of N words of
 * type T, ordered from the most to the least significant word. With N = 2 and
 * T = float, it has about 48 significand bits and the exponent range of float.
 *
 * The algorithms for N = 2 and their relative error bounds, where
 * u = 2^-p is the unit roundoff of T, x and y are multi_words and a is a
 * single word of type T:
 *
 * | operation | accurate              | fast                   | pair      |
 * |-----------|-----------------------|------------------------|-----------|
 * | x + y     | AccurateDWPlusDW,     | SloppyDWPlusDW, 3u^2   | CPairSum  |
 * |           | 3u^2 / (1 - 4u)       | for equal signs only   |           |
 * | x + a     | DWPlusFP, 2u^2        | DWPlusFP, 2u^2         | CPairSum  |
 * | x * y     | DWTimesDW3, 5u^2      | DWTimesDW2, 5u^2 (FMA) | CPairProd |
 * |           | (FMA only)            | DWTimesDW1, 7u^2       |           |
 * | x * a     | DWTimesFP3, 2u^2 (FMA)| DWTimesFP3, 2u^2 (FMA) | CPairProd |
 * |           | DWTimesFP1, 1.5u^2    | DWTimesFP2, 3u^2       |           |
 * | x / y     | DWDivDW3, 9.8u^2      | DWDivDW2, 15u^2        | CPairDiv  |
 * |           | (FMA only)            |                        |           |
 * | x / a     | DWDivFP3, 3u^2        | DWDivFP3, 3u^2         | CPairDiv  |
 *
 * Subtraction adds the exact negation, a / x is multi_word(a) / x. Pair
 * arithmetic (normalization::none) gives the value of the fast algorithm
 * without its final normalization, for a single operation on normalized inputs.
 *
 * Only N = 2 is implemented so far, except for negation, comparison and
 * eval, which work for any N.
 *
 * @tparam T  the type of the words, float or double
 * @tparam N  the number of words
 * @tparam Policy  the compile-time options, see mw::policy
 */
template <typename T, int N, typename Policy = mw::policy<>>
class multi_word {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "multi_word supports float and double words");
    static_assert(N >= 2, "multi_word needs at least two words");

    using words_type = mw::detail::word_array<T, N>;
    using arithmetic = mw::detail::arithmetic<T, N, Policy>;

    // enables mixed operations only for a single word of type T, so that
    // other floating-point types are not silently rounded to T
    template <typename U>
    using enable_if_word = std::enable_if_t<std::is_same_v<U, T>>;

public:
    using word_type = T;
    using policy_type = Policy;
    static constexpr int num_words = N;

    /**
     * Creates zero.
     */
    GKO_ATTRIBUTES constexpr multi_word() noexcept : words_{} {}

    /**
     * Creates the value of a single word or an integer.
     */
    // the condition has to be part of the template parameter's type, unlike
    // typename = std::enable_if_t<...>: otherwise the two constructors from U
    // would be the same template
    template <typename U, std::enable_if_t<std::is_same_v<U, T> ||
                                           std::is_integral_v<U>>* = nullptr>
    GKO_ATTRIBUTES constexpr multi_word(U value) noexcept
        : words_{{static_cast<T>(value)}}
    {}

    /**
     * Rounds a value of another floating-point type to N words: each word is
     * the rest of the value rounded to T. The result is normalized. This is
     * exact if the value fits into the N words, e.g. a double with at most
     * 48 significand bits for two float words.
     *
     * It is explicit, because a double in general does not fit into two
     * float words.
     */
    template <typename U, std::enable_if_t<std::is_floating_point_v<U> &&
                                           !std::is_same_v<U, T>>* = nullptr>
    GKO_ATTRIBUTES explicit multi_word(U value) : words_{round_to_words(value)}
    {}

    /**
     * Creates a number from its N words as they are, without normalizing them.
     * With eager normalization, the operations expect normalized words.
     *
     * @param words  the words, the most significant one first
     */
    template <typename... Words,
              typename = std::enable_if_t<sizeof...(Words) == N &&
                                          (std::is_same_v<Words, T> && ...)>>
    static GKO_ATTRIBUTES constexpr multi_word from_words(
        Words... words) noexcept
    {
        return multi_word{words_type{{words...}}};
    }

    /**
     * @return the i-th word, starting from the most significant one
     */
    GKO_ATTRIBUTES constexpr T word(int i) const noexcept { return words_[i]; }

    /**
     * Negation is exact.
     */
    GKO_ATTRIBUTES constexpr multi_word operator-() const noexcept
    {
        words_type result{};
#pragma unroll
        for (int i = 0; i < N; ++i) {
            result[i] = -words_[i];
        }
        return multi_word{result};
    }

    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator+(const multi_word& x,
                                                          const multi_word& y)
    {
        return multi_word{checked(arithmetic::add(x.words_, y.words_), [&] {
            return mw::detail::add_rn(x.words_[0], y.words_[0]);
        })};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator+(const multi_word& x,
                                                          U a)
    {
        return multi_word{checked(arithmetic::add(x.words_, a), [&] {
            return mw::detail::add_rn(x.words_[0], a);
        })};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator+(U a,
                                                          const multi_word& x)
    {
        return x + a;
    }

    // TODO are there faster subtraction algorithms?
    /**
     * Subtraction is the addition of the exact negation.
     */
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator-(const multi_word& x,
                                                          const multi_word& y)
    {
        return x + (-y);
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator-(const multi_word& x,
                                                          U a)
    {
        return x + (-a);
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator-(U a,
                                                          const multi_word& x)
    {
        return (-x) + a;
    }

    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator*(const multi_word& x,
                                                          const multi_word& y)
    {
        return multi_word{checked(arithmetic::mul(x.words_, y.words_), [&] {
            return mw::detail::mul_rn(x.words_[0], y.words_[0]);
        })};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator*(const multi_word& x,
                                                          U a)
    {
        return multi_word{checked(arithmetic::mul(x.words_, a), [&] {
            return mw::detail::mul_rn(x.words_[0], a);
        })};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator*(U a,
                                                          const multi_word& x)
    {
        return x * a;
    }

    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator/(const multi_word& x,
                                                          const multi_word& y)
    {
        return multi_word{checked(arithmetic::div(x.words_, y.words_), [&] {
            return mw::detail::div_rn(x.words_[0], y.words_[0]);
        })};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator/(const multi_word& x,
                                                          U a)
    {
        return multi_word{checked(arithmetic::div(x.words_, a), [&] {
            return mw::detail::div_rn(x.words_[0], a);
        })};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator/(U a,
                                                          const multi_word& x)
    {
        return multi_word{a} / x;
    }

    template <typename Other>
    GKO_ATTRIBUTES GKO_INLINE multi_word& operator+=(const Other& y)
    {
        return *this = *this + y;
    }

    template <typename Other>
    GKO_ATTRIBUTES GKO_INLINE multi_word& operator-=(const Other& y)
    {
        return *this = *this - y;
    }

    template <typename Other>
    GKO_ATTRIBUTES GKO_INLINE multi_word& operator*=(const Other& y)
    {
        return *this = *this * y;
    }

    template <typename Other>
    GKO_ATTRIBUTES GKO_INLINE multi_word& operator/=(const Other& y)
    {
        return *this = *this / y;
    }

    /**
     * Returns the same value with normalized words: each word is the sum of
     * itself and all less significant words, rounded to nearest. This is the
     * representation the eager normalization keeps after every operation. In
     * pair arithmetic, it lets the kernels choose when to normalize, e.g. once
     * per iteration.
     *
     * With eager normalization, this uses Fast2Sum, so the leading word must
     * have the largest exponent, as it has after every operation. Pair
     * arithmetic uses TwoSum, which normalizes any words.
     */
    friend GKO_ATTRIBUTES GKO_INLINE multi_word normalize(const multi_word& x)
    {
        return multi_word{normalized_words(x.words_)};
    }

    friend GKO_ATTRIBUTES GKO_INLINE multi_word abs(const multi_word& x)
    {
        return x < zero<multi_word>() ? -x : x;
    }

    /*
     * The comparisons are lexicographic: the first word that differs decides.
     * For normalized words, this is the order of the values. Pair arithmetic
     * normalizes its operands first. NaN compares unequal to everything.
     */

    friend GKO_ATTRIBUTES GKO_INLINE bool operator==(const multi_word& x,
                                                     const multi_word& y)
    {
        const auto a = comparable_words(x.words_);
        const auto b = comparable_words(y.words_);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    friend GKO_ATTRIBUTES GKO_INLINE bool operator!=(const multi_word& x,
                                                     const multi_word& y)
    {
        return !(x == y);
    }

    friend GKO_ATTRIBUTES GKO_INLINE bool operator<(const multi_word& x,
                                                    const multi_word& y)
    {
        return less(comparable_words(x.words_), comparable_words(y.words_),
                    false);
    }

    friend GKO_ATTRIBUTES GKO_INLINE bool operator<=(const multi_word& x,
                                                     const multi_word& y)
    {
        return less(comparable_words(x.words_), comparable_words(y.words_),
                    true);
    }

    friend GKO_ATTRIBUTES GKO_INLINE bool operator>(const multi_word& x,
                                                    const multi_word& y)
    {
        return y < x;
    }

    friend GKO_ATTRIBUTES GKO_INLINE bool operator>=(const multi_word& x,
                                                     const multi_word& y)
    {
        return y <= x;
    }

private:
    GKO_ATTRIBUTES constexpr explicit multi_word(
        const words_type& words) noexcept
        : words_{words}
    {}

    /**
     * Applies the special value policy to the result of an operation.
     *
     * @param leading  computes the operation on the leading words only. It is
     *                 only called for special::handle and a result that is
     *                 not finite, so it costs nothing otherwise.
     */
    template <typename Leading>
    static GKO_ATTRIBUTES GKO_INLINE words_type
    checked(const words_type& result, Leading leading)
    {
        if constexpr (Policy::special_values == mw::special::handle) {
            constexpr bool normalized =
                Policy::norm == mw::normalization::eager;
            return mw::detail::handle_special_values<normalized>(result,
                                                                 leading);
        } else {
            return result;
        }
    }

    static GKO_ATTRIBUTES GKO_INLINE words_type
    normalized_words(const words_type& words)
    {
        return checked(arithmetic::normalize(words),
                       [&] { return mw::detail::add_rn(words[0], words[1]); });
    }

    template <typename U>
    static GKO_ATTRIBUTES GKO_INLINE words_type round_to_words(U value)
    {
        words_type words{};
#pragma unroll
        for (int i = 0; i < N; ++i) {
            words[i] = static_cast<T>(value);
            // exact if U has at least as many significand bits as T,
            // otherwise the rest is zero
            value -= static_cast<U>(words[i]);
        }
        // the words can sum up to a tie that rounds away from the leading
        // word
        return checked(arithmetic::normalize_ordered(words),
                       [&] { return words[0]; });
    }

    static GKO_ATTRIBUTES GKO_INLINE words_type
    comparable_words(const words_type& words)
    {
        if constexpr (Policy::norm == mw::normalization::none) {
            return normalized_words(words);
        } else {
            return words;
        }
    }

    static GKO_ATTRIBUTES GKO_INLINE bool less(const words_type& x,
                                               const words_type& y,
                                               bool or_equal)
    {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            // also true if one of them is NaN, which then compares false
            if (x[i] != y[i]) {
                return x[i] < y[i];
            }
        }
        return or_equal;
    }

    words_type words_;
};


namespace detail {


// a real scalar type, e.g. for remove_complex and gko::abs
template <typename T, int N, typename Policy>
struct is_complex_or_scalar_impl<multi_word<T, N, Policy>> : std::true_type {};


}  // namespace detail


namespace mw {


/**
 * Evaluates the sum of the words in type U, usually a type with more
 * precision than the words, e.g. double for float words. The words are added
 * from the least to the most significant one. For N = 2, this rounds only
 * once, so for two float words it gives the double nearest to the value.
 */
template <typename U, typename T, int N, typename Policy>
GKO_ATTRIBUTES constexpr U eval(const multi_word<T, N, Policy>& x) noexcept
{
    U sum{};
#pragma unroll
    for (int i = N - 1; i >= 0; --i) {
        sum += static_cast<U>(x.word(i));
    }
    return sum;
}


}  // namespace mw
}  // namespace gko


#endif  // GKO_CORE_BASE_MULTI_WORD_HPP_
