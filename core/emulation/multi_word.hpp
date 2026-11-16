// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_EMULATION_MULTI_WORD_HPP_
#define GKO_CORE_EMULATION_MULTI_WORD_HPP_


#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
/**
 * Compile-time options and building blocks of multi_word: the basic operations
 * rounded to nearest and the error-free transformations. The arithmetic of two
 * words is in core/emulation/double_word.hpp.
 *
 * The algorithm numbers refer to Joldes, Muller, Popescu, "Tight and rigorous
 * error bounds for basic building blocks of double-word arithmetic", 2017,
 * https://doi.org/10.1145/3121432.
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
 * Which algorithm variant the operations use, see
 * core/emulation/double_word.hpp for the algorithms of two words and their
 * error bounds.
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
     * overflow gives inf, as for a single word. This costs a comparison per
     * operation. If an intermediate step overflows or gives NaN although the
     * exact result is finite, e.g. in TwoProd without FMA for words above
     * about max / 2^(p/2), the result only has single-word accuracy.
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
 * The basic operations, rounded to nearest. The algorithms need each of them
 * rounded on its own, with an FMA only where they call fma_rn: a product that
 * the compiler contracts into an FMA breaks them, e.g. Veltkamp's splitting.
 * On the device, the intrinsics prevent the contraction. On the host, GCC
 * contracts whenever the target has FMA instructions, e.g. with -mfma or
 * -march=native on x86-64.
 */


template <typename T>
GKO_ATTRIBUTES GKO_INLINE T add_rn(T a, T b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>) {
        return __fadd_rn(a, b);
    } else if constexpr (std::is_same_v<T, double>) {
        return __dadd_rn(a, b);
    } else {
        static_assert(sizeof(T) == 0, "not implemented");
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
    } else if constexpr (std::is_same_v<T, double>) {
        return __dsub_rn(a, b);
    } else {
        static_assert(sizeof(T) == 0, "not implemented");
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
    } else if constexpr (std::is_same_v<T, double>) {
        return __dmul_rn(a, b);
    } else {
        static_assert(sizeof(T) == 0, "not implemented");
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
    } else if constexpr (std::is_same_v<T, double>) {
        return __ddiv_rn(a, b);
    } else {
        static_assert(sizeof(T) == 0, "not implemented");
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
    } else if constexpr (std::is_same_v<T, double>) {
        return __fma_rn(a, b, c);
    } else {
        static_assert(sizeof(T) == 0, "not implemented");
    }
#else
    return std::fma(a, b, c);
#endif
}


/**
 * Returns the square root, correctly rounded.
 */
template <typename T>
GKO_ATTRIBUTES GKO_INLINE T sqrt_rn(T a)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same_v<T, float>) {
        return __fsqrt_rn(a);
    } else if constexpr (std::is_same_v<T, double>) {
        return __dsqrt_rn(a);
    } else {
        static_assert(sizeof(T) == 0, "not implemented");
    }
#else
    return std::sqrt(a);
#endif
}


/**
 * FloatType for gko::truncated<FloatType, ...>, U otherwise.
 */
template <typename U>
struct untruncated {
    using type = U;
};

template <typename FloatType, size_type NumComponents, size_type ComponentId>
struct untruncated<truncated<FloatType, NumComponents, ComponentId>> {
    using type = FloatType;
};


/**
 * Whether U is a type like half or bfloat16 for words of type T: smaller than
 * T, not a C++ arithmetic type, and convertible to and from T. gko::truncated
 * counts as its FloatType, so it is none; it has its own overloads in
 * multi_word.
 *
 * @note It is at namespace scope, since nvcc ignores conversion functions
 *       whose enable_if uses a private member.
 */
template <typename U, typename T>
constexpr bool is_small_word_type_v =
    !std::is_arithmetic_v<typename untruncated<U>::type> &&
    sizeof(U) < sizeof(T) && std::is_constructible_v<T, U> &&
    std::is_constructible_v<U, T>;


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


template <typename T>
GKO_ATTRIBUTES GKO_INLINE bool is_finite(T value)
{
    constexpr auto inf = std::numeric_limits<T>::infinity();
    return value > -inf && value < inf;
}


/**
 * Whether the result of an operation is finite, for special::handle.
 *
 * @tparam Normalized  whether the result is normalized. Then a lower word can
 *                     only be infinite or NaN if the leading word is, so only
 *                     the leading word is checked.
 */
template <bool Normalized, typename T, int N>
GKO_ATTRIBUTES GKO_INLINE bool is_finite_result(const word_array<T, N>& result)
{
    bool finite = is_finite(result[0]);
    if constexpr (!Normalized) {
#pragma unroll
        for (int i = 1; i < N; ++i) {
            finite = finite && is_finite(result[i]);
        }
    }
    return finite;
}


/**
 * The result of an operation for special::handle, if is_finite_result is
 * false: the same operation on the leading words only, with zero lower words,
 * as for a single word.
 *
 * @param leading  the operation on the leading words
 */
template <typename T, int N>
GKO_ATTRIBUTES GKO_INLINE word_array<T, N> special_result(T leading)
{
    constexpr auto inf = std::numeric_limits<T>::infinity();
    constexpr auto near_overflow =
        std::numeric_limits<T>::max() *
        (T{1} - 4 * std::numeric_limits<T>::epsilon());
    word_array<T, N> result{};
    if (leading >= near_overflow) {
        result[0] = inf;
    } else if (leading <= -near_overflow) {
        result[0] = -inf;
    } else {
        result[0] = leading;
    }
    return result;
}


/**
 * Selects the algorithms for N words and a policy, and applies the special
 * value policy with is_finite_result and special_result. Each N needs its own
 * algorithms, so each N specializes this class, see e.g.
 * core/emulation/double_word.hpp for N = 2.
 */
template <typename T, int N, typename Policy>
struct arithmetic {
    // false for every type, so it fails for every N without a specialization
    static_assert(
        sizeof(T) == 0,
        "the arithmetic of two words is in "
        "core/emulation/double_word.hpp, other N are not yet implemented");
};


}  // namespace detail
}  // namespace mw


template <typename T, int N, typename Policy>
class multi_word;


namespace mw {


/**
 * Evaluates x in type U, usually a type with more precision than the words,
 * e.g. double for float words. The words are added from the least to the most
 * significant one, so if U represents the words exactly, two words round only
 * once.
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


/**
 * A floating-point number represented as the unevaluated sum of N words of
 * type T, ordered from the most to the least significant word. With N = 2 and
 * T = float, it has about 48 significand bits and the exponent range of float.
 *
 * The sign of zero is not kept: e.g. -0.0 converts to +0, and a zero result
 * converts back to +0. Ginkgo does not depend on the sign of zero.
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
     * Leaves the words uninitialized, as for T, so that multi_word is a
     * trivial type, e.g. in __shared__ memory. multi_word{} is zero.
     */
    multi_word() noexcept = default;

    /**
     * Creates the value of a single word or an integer. Integers of at most 32
     * bits are exact, larger ones go through double.
     * Prevents other floating points to be silently casted.
     */
    template <typename U, std::enable_if_t<std::is_same_v<U, T> ||
                                           std::is_integral_v<U>>* = nullptr>
    GKO_ATTRIBUTES constexpr multi_word(U value) noexcept
        : words_{words_of(value)}
    {}

    /**
     * Converts a value of a type like half or bfloat16 through the word type.
     */
    template <
        typename U,
        std::enable_if_t<mw::detail::is_small_word_type_v<U, T>>* = nullptr>
    GKO_ATTRIBUTES explicit multi_word(U value)
        : multi_word{static_cast<T>(value)}
    {}

    /**
     * Converts a value of gko::truncated through its FloatType.
     */
    template <typename FloatType, size_type NumComponents,
              size_type ComponentId>
    GKO_ATTRIBUTES explicit multi_word(
        truncated<FloatType, NumComponents, ComponentId> value)
        : multi_word{static_cast<FloatType>(value)}
    {}

    /**
     * Explicitly rounds a value of another floating-point type to N words: each
     * word is the rest of the value rounded to T. The result is normalized.
     */
    // TODO: on the device, rounding a double to float words uses FP64
    // instructions (F2F, DADD), and so does the conversion back, see mw::eval.
    // Splitting the bits of the double with integer operations avoids them.
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
     * Converts to the floating-point type U, see mw::eval. It is explicit,
     * because it rounds in general, and because implicit conversions would
     * make the mixed operations with single words ambiguous.
     */
    template <typename U,
              std::enable_if_t<std::is_floating_point_v<U>>* = nullptr>
    GKO_ATTRIBUTES constexpr explicit operator U() const noexcept
    {
        return mw::eval<U>(*this);
    }

    /**
     * Converts to the integral type U
     * TODO currently done through double. When dealing with integer conversion
     * fix this
     */
    template <typename U, std::enable_if_t<std::is_integral_v<U> &&
                                           !std::is_same_v<U, bool>>* = nullptr>
    GKO_ATTRIBUTES constexpr explicit operator U() const noexcept
    {
        return static_cast<U>(mw::eval<double>(*this));
    }

    /**
     * Converts to a type like half or bfloat16 through the word type. Due to
     * the double rounding, ties can be off by one ulp.
     */
    template <
        typename U,
        std::enable_if_t<mw::detail::is_small_word_type_v<U, T>>* = nullptr>
    GKO_ATTRIBUTES explicit operator U() const
    {
        return static_cast<U>(mw::eval<T>(*this));
    }

    /**
     * Converts to gko::truncated by rounding to FloatType and truncating.
     */
    template <typename FloatType, size_type NumComponents,
              size_type ComponentId>
    GKO_ATTRIBUTES explicit
    operator truncated<FloatType, NumComponents, ComponentId>() const
    {
        return truncated<FloatType, NumComponents, ComponentId>{
            static_cast<FloatType>(*this)};
    }

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
        return multi_word{arithmetic::add(x.words_, y.words_)};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator+(const multi_word& x,
                                                          U a)
    {
        return multi_word{arithmetic::add(x.words_, a)};
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
        return multi_word{arithmetic::mul(x.words_, y.words_)};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator*(const multi_word& x,
                                                          U a)
    {
        return multi_word{arithmetic::mul(x.words_, a)};
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
        return multi_word{arithmetic::div(x.words_, y.words_)};
    }

    template <typename U, typename = enable_if_word<U>>
    friend GKO_ATTRIBUTES GKO_INLINE multi_word operator/(const multi_word& x,
                                                          U a)
    {
        return multi_word{arithmetic::div(x.words_, a)};
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

    /**
     * The square root. Negative values give NaN, as for a single word.
     */
    friend GKO_ATTRIBUTES GKO_INLINE multi_word sqrt(const multi_word& x)
    {
        return multi_word{arithmetic::sqrt(x.words_)};
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

    static GKO_ATTRIBUTES GKO_INLINE words_type
    normalized_words(const words_type& words)
    {
        return arithmetic::normalize(words);
    }

    // The leading word is the integer rounded to T, the following words hold
    // the rest. The rest is computed in std::int64_t, since the rounded leading
    // word may not fit into U, e.g. 2^31 for the largest int32.
    template <typename U>
    static GKO_ATTRIBUTES constexpr words_type words_of(U value) noexcept
    {
        if constexpr (std::is_same_v<U, T>) {
            return words_type{{value}};
        } else if constexpr (sizeof(U) <= sizeof(std::int32_t)) {
            // TODO when dealing with integer conversions, deal with this
            words_type words{};
            std::int64_t rest = value;
            for (int i = 0; i < N; ++i) {
                words[i] = static_cast<T>(rest);
                rest -= static_cast<std::int64_t>(words[i]);
            }
            return words;
        } else {
            // TODO when dealing with integer conversions, deal with this
            return round_to_words(static_cast<double>(value));
        }
    }

    template <typename U>
    static GKO_ATTRIBUTES GKO_INLINE words_type round_to_words(U value)
    {
        words_type words{};
#pragma unroll
        for (int i = 0; i < N; ++i) {
            words[i] = static_cast<T>(value);
            value -= static_cast<U>(words[i]);
        }
        // the words can sum up to a tie that rounds away from the leading
        // word
        return arithmetic::normalize_ordered(words);
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


/**
 * Whether T is a multi_word.
 */
template <typename T>
struct is_multi_word : std::false_type {};

template <typename T, int N, typename Policy>
struct is_multi_word<multi_word<T, N, Policy>> : std::true_type {};

template <typename T>
constexpr bool is_multi_word_v = is_multi_word<T>::value;


namespace detail {


// a real scalar type, e.g. for remove_complex and gko::abs
template <typename T, int N, typename Policy>
struct is_complex_or_scalar_impl<multi_word<T, N, Policy>> : std::true_type {};

// TODO add numeric_limits specializations to multi_word
// std::numeric_limits is not specialized for multi_word, so its infinity() is
// zero, which would make gko::is_finite false for every value
template <typename T, int N, typename Policy>
struct infinity_impl<multi_word<T, N, Policy>> {
    static constexpr multi_word<T, N, Policy> value{
        std::numeric_limits<T>::infinity()};
};


}  // namespace detail


}  // namespace gko


#endif  // GKO_CORE_EMULATION_MULTI_WORD_HPP_
