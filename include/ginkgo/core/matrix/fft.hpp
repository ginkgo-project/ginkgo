/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_MATRIX_FFT_HPP_
#define GKO_PUBLIC_CORE_MATRIX_FFT_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


/**
 * This LinOp implements a 1D Fourier matrix using the FFT algorithm.
 *
 * It implements forward and inverse DFT.
 *
 * For a power-of-two size n with corresponding root of unity
 * $\omega = e^{-2\pi i / n}$ for forward DFT and $\omega = e^{2 \pi i / n}$
 * for inverse DFT it computes
 *
 *     $$ x_k = \sum_{j=0}^{n-1} \omega^{jk} b_j $$
 *
 * without normalization factors.
 *
 * @ingroup identity
 * @ingroup mat_formats
 * @ingroup LinOp
 */
class Fft : public EnableLinOp<Fft>,
            public EnableCreateMethod<Fft>,
            public Transposable {
    friend class EnablePolymorphicObject<Fft, LinOp>;
    friend class EnableCreateMethod<Fft>;

public:
    using EnableLinOp<Fft>::convert_to;
    using EnableLinOp<Fft>::move_to;

    using value_type = std::complex<double>;
    using transposed_type = Fft;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    bool is_inverse() const;

protected:
    /**
     * Creates an empty Fourier matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Fft(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Fft>(exec), buffer_{exec}, inverse_{}
    {}

    /**
     * Creates an Fourier matrix with the given dimensions.
     *
     * @param size  size of the matrix
     * @param inverse  true to compute an inverse DFT instead of a normal DFT
     */
    Fft(std::shared_ptr<const Executor> exec, size_type size,
        bool inverse = false)
        : EnableLinOp<Fft>(exec, dim<2>{size}), buffer_{exec}, inverse_{inverse}
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    mutable Array<char> buffer_;
    bool inverse_;
};


/**
 * This LinOp implements a 2D Fourier matrix using the FFT algorithm.
 * For indexing purposes, the first dimension is the major axis.
 *
 * It implements complex-to-complex forward and inverse FFT.
 *
 * For a power-of-two sizes $n_1, n_2$ with corresponding root of unity
 * $\omega = e^{-2\pi i / (n_1 n_2)}$ for forward DFT and
 * $\omega = e^{2 \pi i / (n_1 n_2)}$ for inverse DFT it computes
 *
 *     $$ x_{k_1 n_2 + k_2} = \sum_{i_1=0}^{n_1-1} \sum_{i_2=0}^{n_2-1}
 *                           \omega^{i_1 k_1 + i_2 k_2} b_{i_1 n_2 + i_2} $$
 *
 * without normalization factors.
 *
 * @ingroup identity
 * @ingroup mat_formats
 * @ingroup LinOp
 */
class Fft2 : public EnableLinOp<Fft2>,
             public EnableCreateMethod<Fft2>,
             public Transposable {
    friend class EnablePolymorphicObject<Fft2, LinOp>;
    friend class EnableCreateMethod<Fft2>;

public:
    using EnableLinOp<Fft2>::convert_to;
    using EnableLinOp<Fft2>::move_to;

    using value_type = std::complex<double>;
    using transposed_type = Fft2;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    bool is_inverse() const;

protected:
    /**
     * Creates an empty Fourier matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Fft2(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Fft2>(exec), buffer_{exec}, size1_{}, size2_{}, inverse_{}
    {}

    /**
     * Creates an Fourier matrix with the given dimensions.
     *
     * @param size  size of both FFT dimensions
     */
    Fft2(std::shared_ptr<const Executor> exec, size_type size)
        : Fft2{exec, size, size}
    {}

    /**
     * Creates an Fourier matrix with the given dimensions.
     *
     * @param size1  size of the first FFT dimension
     * @param size2  size of the second FFT dimension
     * @param inverse  true to compute an inverse DFT instead of a normal DFT
     */
    Fft2(std::shared_ptr<const Executor> exec, size_type size1, size_type size2,
         bool inverse = false)
        : EnableLinOp<Fft2>(exec, dim<2>{size1 * size2}),
          buffer_{exec},
          size1_{size1},
          size2_{size2},
          inverse_{inverse}
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    mutable Array<char> buffer_;
    size_type size1_;
    size_type size2_;
    bool inverse_;
};


/**
 * This LinOp implements a 3D Fourier matrix using the FFT algorithm.
 * For indexing purposes, the first dimension is the major axis.
 *
 * It implements complex-to-complex forward and inverse FFT.
 *
 * For a power-of-two sizes $n_1, n_2, n_3$ with corresponding root of unity
 * $\omega = e^{-2\pi i / (n_1 n_2 n_3)}$ for forward DFT and
 * $\omega = e^{2 \pi i / (n_1 n_2 n_3)}$ for inverse DFT it computes
 *
 *     $$ x_{k_1 n_2 n_3 + k_2 n_3 + k_3} = \sum_{i_1=0}^{n_1-1}
 *                     \sum_{i_2=0}^{n_2-1} \sum_{i_3=0}^{n_3-1}
 *                     \omega^{i_1 k_1 + i_2 k_2 + i_3 k_3}
 *                     b_{i_1 n_2 n_3 + i_2 n_3 + i_3} $$
 *
 * without normalization factors.
 *
 * @ingroup identity
 * @ingroup mat_formats
 * @ingroup LinOp
 */
class Fft3 : public EnableLinOp<Fft3>,
             public EnableCreateMethod<Fft3>,
             public Transposable {
    friend class EnablePolymorphicObject<Fft3, LinOp>;
    friend class EnableCreateMethod<Fft3>;

public:
    using EnableLinOp<Fft3>::convert_to;
    using EnableLinOp<Fft3>::move_to;

    using value_type = std::complex<double>;
    using transposed_type = Fft3;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    bool is_inverse() const;

protected:
    /**
     * Creates an empty Fourier matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Fft3(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Fft3>(exec),
          buffer_{exec},
          size1_{},
          size2_{},
          size3_{},
          inverse_{}
    {}

    /**
     * Creates an Fourier matrix with the given dimensions.
     *
     * @param size  size of all FFT dimensions
     */
    Fft3(std::shared_ptr<const Executor> exec, size_type size)
        : Fft3{exec, size, size, size}
    {}

    /**
     * Creates an Fourier matrix with the given dimensions.
     *
     * @param size1  size of the first FFT dimension
     * @param size2  size of the second FFT dimension
     * @param size3  size of the third FFT dimension
     * @param inverse  true to compute an inverse DFT instead of a normal DFT
     */
    Fft3(std::shared_ptr<const Executor> exec, size_type size1, size_type size2,
         size_type size3, bool inverse = false)
        : EnableLinOp<Fft3>(exec, dim<2>{size1 * size2 * size3}),
          buffer_{exec},
          size1_{size1},
          size2_{size2},
          size3_{size3},
          inverse_{inverse}
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    mutable Array<char> buffer_;
    size_type size1_;
    size_type size2_;
    size_type size3_;
    bool inverse_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_FFT_HPP_
