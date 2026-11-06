// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_EMULATION_DOUBLE_WORD_HPP_
#define GKO_CORE_EMULATION_DOUBLE_WORD_HPP_


#include <limits>

#include "core/emulation/multi_word.hpp"


namespace gko {
/**
 * The arithmetic of multi_word with two words: double-word arithmetic, which
 * normalizes after every operation, and pair arithmetic, which does not.
 *
 * The algorithms and their relative error bounds, where u = 2^-p is the unit
 * roundoff of T, x and y are multi_words and a is a single word of type T:
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
 * | sqrt(x)   | SQRTDWtoDW, 25/8 u^2  | SQRTDWtoDW, 25/8 u^2   | see below |
 *
 * The bound of SQRTDWtoDW is the one of Lefevre et al. 2023 with FMA; without
 * FMA, the same algorithm computes the same exact remainder with Dekker's
 * product. Pair arithmetic normalizes the operand of sqrt and skips the final
 * Fast2Sum. Subtraction adds the exact negation, a / x is multi_word(a) / x.
 * Pair arithmetic (normalization::none) gives the value of the fast algorithm
 * without its final normalization, for a single operation on normalized inputs.
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
 * - Lefevre, Louvet, Muller, Picot, Rideau, "Accurate calculation of
 *   Euclidean norms using double-word arithmetic", ACM TOMS, 2023
 *   (SQRTDWtoDW)
 * - Noack, "twofloat", https://github.com/esa-tu-darmstadt/twofloat, which
 *   implements both with the same algorithms.
 */
namespace mw {
namespace detail {


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


/**
 * The square root of a normalized x with 0 < x[0] < inf, before the final
 * normalization: SQRTDWtoDW (Lefevre et al. 2023) without its Fast2Sum. The
 * leading word is the square root of x[0], the second word corrects it by
 * (x - s^2) / (2 s), where x[0] - s^2 is exact: the remainder of a correctly
 * rounded square root is a floating-point number. Without FMA, Dekker's
 * product gives s^2 exactly.
 */
template <bool Fma, typename T>
GKO_ATTRIBUTES GKO_INLINE word_array<T, 2> dw_sqrt(const word_array<T, 2>& x)
{
    const T s = sqrt_rn(x[0]);
    T remainder{};
    if constexpr (Fma) {
        remainder = fma_rn(-s, s, x[0]);
    } else {
        // s^2 is within a factor of two of x[0], so x[0] - p[0] is exact
        const auto p = two_prod<false>(s, s);
        remainder = sub_rn(sub_rn(x[0], p[0]), p[1]);
    }
    return {{s, div_rn(add_rn(remainder, x[1]), add_rn(s, s))}};
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


template <typename T, typename Policy>
struct arithmetic<T, 2, Policy> {
    using words = word_array<T, 2>;

    static constexpr bool pair = Policy::norm == normalization::none;
    static constexpr bool fast = Policy::alg == algorithm::fast;
    static constexpr bool fma = Policy::fma;
    // special::handle replaces results that are not finite by the operation
    // on the leading words, see is_finite_result and special_result
    static constexpr bool handle = Policy::special_values == special::handle;
    // only the eager normalization keeps the results normalized
    static constexpr bool normalized = !pair;

    /**
     * Normalizes words whose leading word has the largest exponent, e.g. the
     * words of a value rounded word by word. For them, Fast2Sum is exact.
     */
    static GKO_ATTRIBUTES GKO_INLINE words normalize_ordered(const words& x)
    {
        words result = fast_two_sum(x[0], x[1]);
        if constexpr (handle) {
            if (!is_finite_result<normalized>(result)) {
                result = special_result<T, 2>(x[0]);
            }
        }
        return result;
    }

    /**
     * Normalizes the words of a result. The double-word algorithms keep the
     * leading word the largest, so Fast2Sum suffices. Pair arithmetic can
     * leave a second word with a larger exponent, which needs TwoSum.
     */
    static GKO_ATTRIBUTES GKO_INLINE words normalize(const words& x)
    {
        words result;
        if constexpr (pair) {
            result = two_sum(x[0], x[1]);
        } else {
            result = fast_two_sum(x[0], x[1]);
        }
        if constexpr (handle) {
            if (!is_finite_result<normalized>(result)) {
                result = special_result<T, 2>(add_rn(x[0], x[1]));
            }
        }
        return result;
    }

    static GKO_ATTRIBUTES GKO_INLINE words add(const words& x, const words& y)
    {
        words result;
        if constexpr (pair) {
            result = pair_sum(x, y);
        } else if constexpr (fast) {
            result = sloppy_dw_plus_dw(x, y);
        } else {
            result = accurate_dw_plus_dw(x, y);
        }
        if constexpr (handle) {
            if (!is_finite_result<normalized>(result)) {
                result = special_result<T, 2>(add_rn(x[0], y[0]));
            }
        }
        return result;
    }

    static GKO_ATTRIBUTES GKO_INLINE words add(const words& x, T a)
    {
        words result;
        if constexpr (pair) {
            result = pair_sum_fp(x, a);
        } else {
            result = dw_plus_fp(x, a);
        }
        if constexpr (handle) {
            if (!is_finite_result<normalized>(result)) {
                result = special_result<T, 2>(add_rn(x[0], a));
            }
        }
        return result;
    }

    static GKO_ATTRIBUTES GKO_INLINE words mul(const words& x, const words& y)
    {
        static_assert(pair || fma || fast,
                      "the accurate product of two multi_words needs FMA "
                      "(DWTimesDW3), use algorithm::fast without FMA");
        words result;
        if constexpr (pair) {
            result = pair_prod<fma>(x, y);
        } else if constexpr (fma && fast) {
            result = dw_times_dw2(x, y);
        } else if constexpr (fma) {
            result = dw_times_dw3(x, y);
        } else {
            result = dw_times_dw1<false>(x, y);
        }
        if constexpr (handle) {
            if (!is_finite_result<normalized>(result)) {
                result = special_result<T, 2>(mul_rn(x[0], y[0]));
            }
        }
        return result;
    }

    static GKO_ATTRIBUTES GKO_INLINE words mul(const words& x, T a)
    {
        words result;
        if constexpr (pair) {
            result = pair_prod_fp<fma>(x, a);
        } else if constexpr (fma) {
            result = dw_times_fp3(x, a);
        } else if constexpr (fast) {
            result = dw_times_fp2<false>(x, a);
        } else {
            result = dw_times_fp1<false>(x, a);
        }
        if constexpr (handle) {
            if (!is_finite_result<normalized>(result)) {
                result = special_result<T, 2>(mul_rn(x[0], a));
            }
        }
        return result;
    }

    static GKO_ATTRIBUTES GKO_INLINE words div(const words& x, const words& y)
    {
        static_assert(pair || fma || fast,
                      "the accurate quotient of two multi_words needs FMA "
                      "(DWDivDW3), use algorithm::fast without FMA");
        words result;
        if constexpr (pair) {
            result = pair_div<fma>(x, y);
        } else if constexpr (fast) {
            result = dw_div_dw2<fma>(x, y);
        } else {
            result = dw_div_dw3(x, y);
        }
        if constexpr (handle) {
            if (!is_finite_result<normalized>(result)) {
                result = special_result<T, 2>(div_rn(x[0], y[0]));
            }
        }
        return result;
    }

    static GKO_ATTRIBUTES GKO_INLINE words div(const words& x, T a)
    {
        words result;
        if constexpr (pair) {
            result = pair_div_fp<fma>(x, a);
        } else {
            result = dw_div_fp3<fma>(x, a);
        }
        if constexpr (handle) {
            if (!is_finite_result<normalized>(result)) {
                result = special_result<T, 2>(div_rn(x[0], a));
            }
        }
        return result;
    }

    /**
     * Zero, negative, infinite and NaN values give the square root of the
     * leading word with a zero second word, as for a single word, for any
     * special value policy: sqrt(0) has to stay 0, and the algorithm would
     * divide 0 by 0. Pair arithmetic normalizes with TwoSum first, because
     * its leading word can be zero while the value is not.
     */
    static GKO_ATTRIBUTES GKO_INLINE words sqrt(const words& x)
    {
        const words n = pair ? two_sum(x[0], x[1]) : x;
        if (!(n[0] > T{} && n[0] < std::numeric_limits<T>::infinity())) {
            return {{sqrt_rn(n[0]), T{}}};
        }
        const auto root = dw_sqrt<fma>(n);
        if constexpr (pair) {
            return root;
        } else {
            return fast_two_sum(root[0], root[1]);
        }
    }
};


}  // namespace detail
}  // namespace mw
}  // namespace gko


#endif  // GKO_CORE_EMULATION_DOUBLE_WORD_HPP_
