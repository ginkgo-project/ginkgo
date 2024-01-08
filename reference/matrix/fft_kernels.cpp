// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fft_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The FFT matrix format namespace.
 * @ref Fft
 * @ingroup fft
 */
namespace fft {


template <typename InValueType, typename OutValueType>
void bfly(const matrix::Dense<InValueType>* b, matrix::Dense<OutValueType>* x,
          int64 lo, int64 hi, OutValueType root)
{
    for (size_type rhs = 0; rhs < x->get_size()[1]; rhs++) {
        auto lo_val = b->at(lo, rhs);
        auto hi_val = b->at(hi, rhs);
        x->at(lo, rhs) = lo_val + hi_val;
        x->at(hi, rhs) = (lo_val - hi_val) * root;
    }
}


template <typename ValueType>
void bfly(matrix::Dense<ValueType>* x, int64 lo, int64 hi, ValueType root)
{
    for (size_type rhs = 0; rhs < x->get_size()[1]; rhs++) {
        auto lo_val = x->at(lo, rhs);
        auto hi_val = x->at(hi, rhs);
        x->at(lo, rhs) = lo_val + hi_val;
        x->at(hi, rhs) = (lo_val - hi_val) * root;
    }
}


template <typename ValueType>
void bit_rev_swap(matrix::Dense<ValueType>* x, int64 i, int64 rev_i)
{
    for (size_type rhs = 0; rhs < x->get_size()[1]; rhs++) {
        if (i < rev_i) {
            std::swap(x->at(i, rhs), x->at(rev_i, rhs));
        }
    }
}


int64 bit_rev(int64 i, int64 size)
{
    int64 rev{};
    for (int64 fwd = 1, bwd = size / 2; fwd < size; fwd *= 2, bwd /= 2) {
        rev |= ((i / fwd) & 1) * bwd;
    }
    return rev;
}


template <typename ValueType>
vector<ValueType> build_unit_roots(std::shared_ptr<const DefaultExecutor> exec,
                                   int64 size, int64 sign)
{
    vector<ValueType> roots(size / 2, {exec});
    for (int64 i = 0; i < size / 2; i++) {
        roots[i] = unit_root<ValueType>(size, sign * i);
    }
    return roots;
}


template <typename ValueType>
void fft(std::shared_ptr<const DefaultExecutor> exec,
         const matrix::Dense<std::complex<ValueType>>* b,
         matrix::Dense<std::complex<ValueType>>* x, bool inverse,
         array<char>& buffer)
{
    using complex_type = std::complex<ValueType>;
    using real_type = ValueType;
    const int64 sign = inverse ? 1 : -1;
    const auto nrhs = b->get_size()[1];
    const auto size = static_cast<int64>(b->get_size()[0]);
    GKO_ASSERT_IS_POWER_OF_TWO(size);
    auto roots = build_unit_roots<complex_type>(exec, size, sign);
    // first butterfly step
    auto d = size / 2;
    for (int64 k = 0; k < size / 2; k++) {
        bfly(b, x, k, k + d, roots[k]);
    }
    // remaining butterfly steps
    for (d = size / 4; d > 0; d /= 2) {
        // square roots
        for (int64 i = 0; i < d; i++) {
            roots[i] = roots[2 * i];
        }
        for (int64 base = 0; base < size; base += d * 2) {
            for (int64 k = base; k < base + d; k++) {
                bfly(x, k, k + d, roots[k - base]);
            }
        }
    }
    // bit reversal
    for (int64 k = 0; k < size; k++) {
        bit_rev_swap(x, k, bit_rev(k, size));
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT_KERNEL);


template <typename ValueType>
void fft2(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Dense<std::complex<ValueType>>* b,
          matrix::Dense<std::complex<ValueType>>* x, size_type size1,
          size_type size2, bool inverse, array<char>& buffer)
{
    using complex_type = std::complex<ValueType>;
    using real_type = ValueType;
    const int64 sign = inverse ? 1 : -1;
    const auto nrhs = b->get_size()[1];
    const auto ssize1 = static_cast<int64>(size1);
    const auto ssize2 = static_cast<int64>(size2);
    GKO_ASSERT_IS_POWER_OF_TWO(ssize1);
    GKO_ASSERT_IS_POWER_OF_TWO(ssize2);
    const auto idx = [&](int64 x, int64 y) { return x * ssize2 + y; };
    auto roots1 = build_unit_roots<complex_type>(exec, ssize1, sign);
    auto roots2 = build_unit_roots<complex_type>(exec, ssize2, sign);
    // minor dimension first: size1 FFTs
    // first butterfly step
    auto d = ssize2 / 2;
    for (int64 k1 = 0; k1 < ssize1; k1++) {
        for (int64 k2 = 0; k2 < ssize2 / 2; k2++) {
            bfly(b, x, idx(k1, k2), idx(k1, k2 + d), roots2[k2]);
        }
    }
    // remaining butterfly steps
    for (d = ssize2 / 4; d > 0; d /= 2) {
        // square roots
        for (int64 i = 0; i < d; i++) {
            roots2[i] = roots2[2 * i];
        }
        for (int64 k1 = 0; k1 < ssize1; k1++) {
            for (int64 base = 0; base < ssize2; base += d * 2) {
                for (int64 k2 = base; k2 < base + d; k2++) {
                    bfly(x, idx(k1, k2), idx(k1, k2 + d), roots2[k2 - base]);
                }
            }
        }
    }
    // then major dimension: 1 ssize2-wide FFT
    // butterfly steps
    for (d = ssize1 / 2; d > 0; d /= 2) {
        for (int64 base = 0; base < ssize1; base += d * 2) {
            for (int64 k1 = base; k1 < base + d; k1++) {
                auto root = roots1[k1 - base];
                for (int64 k2 = 0; k2 < ssize2; k2++) {
                    bfly(x, idx(k1, k2), idx(k1 + d, k2), root);
                }
            }
        }
        // square roots
        for (int64 i = 0; i < d / 2; i++) {
            roots1[i] = roots1[2 * i];
        }
    }
    // bit reversal
    for (int64 k1 = 0; k1 < ssize1; k1++) {
        for (int64 k2 = 0; k2 < ssize2; k2++) {
            bit_rev_swap(x, idx(k1, k2),
                         idx(bit_rev(k1, ssize1), bit_rev(k2, ssize2)));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT2_KERNEL);


template <typename ValueType>
void fft3(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Dense<std::complex<ValueType>>* b,
          matrix::Dense<std::complex<ValueType>>* x, size_type size1,
          size_type size2, size_type size3, bool inverse, array<char>& buffer)
{
    using complex_type = std::complex<ValueType>;
    using real_type = ValueType;
    const int64 sign = inverse ? 1 : -1;
    const auto nrhs = b->get_size()[1];
    const auto ssize1 = static_cast<int64>(size1);
    const auto ssize2 = static_cast<int64>(size2);
    const auto ssize3 = static_cast<int64>(size3);
    GKO_ASSERT_IS_POWER_OF_TWO(ssize1);
    GKO_ASSERT_IS_POWER_OF_TWO(ssize2);
    GKO_ASSERT_IS_POWER_OF_TWO(ssize3);
    const auto idx = [&](int64 x, int64 y, int64 z) {
        return x * ssize2 * ssize3 + y * ssize3 + z;
    };
    auto roots1 = build_unit_roots<complex_type>(exec, ssize1, sign);
    auto roots2 = build_unit_roots<complex_type>(exec, ssize2, sign);
    auto roots3 = build_unit_roots<complex_type>(exec, ssize3, sign);
    // minor dimension first: size1*size2 FFTs
    // first butterfly step
    auto d = ssize3 / 2;
    for (int64 k1 = 0; k1 < ssize1; k1++) {
        for (int64 k2 = 0; k2 < ssize2; k2++) {
            for (int64 k3 = 0; k3 < ssize3 / 2; k3++) {
                bfly(b, x, idx(k1, k2, k3), idx(k1, k2, k3 + d), roots3[k3]);
            }
        }
    }
    // remaining butterfly steps
    for (d = ssize3 / 4; d > 0; d /= 2) {
        // square roots
        for (int64 i = 0; i < d; i++) {
            roots3[i] = roots3[2 * i];
        }
        for (int64 k1 = 0; k1 < ssize1; k1++) {
            for (int64 k2 = 0; k2 < ssize2; k2++) {
                for (int64 base = 0; base < ssize3; base += d * 2) {
                    for (int64 k3 = base; k3 < base + d; k3++) {
                        bfly(x, idx(k1, k2, k3), idx(k1, k2, k3 + d),
                             roots3[k3 - base]);
                    }
                }
            }
        }
    }
    // then second dimension: 1 ssize1*ssize3-wide FFT
    // butterfly steps
    for (d = ssize2 / 2; d > 0; d /= 2) {
        for (int64 base = 0; base < ssize2; base += d * 2) {
            for (int64 k2 = base; k2 < base + d; k2++) {
                auto root = roots2[k2 - base];
                for (int64 k1 = 0; k1 < ssize1; k1++) {
                    for (int64 k3 = 0; k3 < ssize3; k3++) {
                        bfly(x, idx(k1, k2, k3), idx(k1, k2 + d, k3), root);
                    }
                }
            }
        }
        // square roots
        for (int64 i = 0; i < d / 2; i++) {
            roots2[i] = roots2[2 * i];
        }
    }
    // then major dimension: 1 ssize2*ssize3-wide FFT
    // butterfly steps
    for (d = ssize1 / 2; d > 0; d /= 2) {
        for (int64 base = 0; base < ssize1; base += d * 2) {
            for (int64 k1 = base; k1 < base + d; k1++) {
                auto root = roots1[k1 - base];
                for (int64 k2 = 0; k2 < ssize2; k2++) {
                    for (int64 k3 = 0; k3 < ssize3; k3++) {
                        bfly(x, idx(k1, k2, k3), idx(k1 + d, k2, k3), root);
                    }
                }
            }
        }
        // square roots
        for (int64 i = 0; i < d / 2; i++) {
            roots1[i] = roots1[2 * i];
        }
    }
    // bit reversal
    for (int64 k1 = 0; k1 < ssize1; k1++) {
        for (int64 k2 = 0; k2 < ssize2; k2++) {
            for (int64 k3 = 0; k3 < ssize3; k3++) {
                bit_rev_swap(x, idx(k1, k2, k3),
                             idx(bit_rev(k1, ssize1), bit_rev(k2, ssize2),
                                 bit_rev(k3, ssize3)));
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT3_KERNEL);


}  // namespace fft
}  // namespace reference
}  // namespace kernels
}  // namespace gko
