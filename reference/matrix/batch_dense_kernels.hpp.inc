/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

template <typename ValueType>
inline void matvec_kernel(
    const gko::batch_dense::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const gko::batch_dense::BatchEntry<ValueType>& c)
{
    for (int row = 0; row < c.num_rows; ++row) {
        for (int col = 0; col < c.num_rhs; ++col) {
            c.values[row * c.stride + col] = gko::zero<ValueType>();
        }
    }

    for (int row = 0; row < c.num_rows; ++row) {
        for (int inner = 0; inner < a.num_rhs; ++inner) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] +=
                    a.values[row * a.stride + inner] *
                    b.values[inner * b.stride + col];
            }
        }
    }
}


template <typename ValueType>
inline void advanced_matvec_kernel(
    const ValueType alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& c)
{
    if (beta != gko::zero<ValueType>()) {
        for (int row = 0; row < c.num_rows; ++row) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] *= beta;
            }
        }
    } else {
        for (int row = 0; row < c.num_rows; ++row) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] *= gko::zero<ValueType>();
            }
        }
    }

    for (int row = 0; row < c.num_rows; ++row) {
        for (int inner = 0; inner < a.num_rhs; ++inner) {
            for (int col = 0; col < c.num_rhs; ++col) {
                c.values[row * c.stride + col] +=
                    alpha * a.values[row * a.stride + inner] *
                    b.values[inner * b.stride + col];
            }
        }
    }
}


template <typename ValueType>
inline void scale(const gko::batch_dense::BatchEntry<const ValueType>& alpha,
                  const gko::batch_dense::BatchEntry<ValueType>& x)
{
    if (alpha.num_rhs == 1) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                x.values[i * x.stride + j] *= alpha.values[0];
            }
        }
    } else {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                x.values[i * x.stride + j] *= alpha.values[j];
            }
        }
    }
}


template <typename ValueType>
inline void add_scaled(
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    if (alpha.num_rhs == 1) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[0] * x.values[i * x.stride + j];
            }
        }
    } else {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] +=
                    alpha.values[j] * x.values[i * x.stride + j];
            }
        }
    }
}


template <typename ValueType>
inline void add_scale(
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<const ValueType>& beta,
    const gko::batch_dense::BatchEntry<ValueType>& y)
{
    if (alpha.num_rhs == 1) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] =
                    alpha.values[0] * x.values[i * x.stride + j] +
                    beta.values[0] * y.values[i * y.stride + j];
            }
        }
    } else {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                y.values[i * y.stride + j] =
                    alpha.values[j] * x.values[i * x.stride + j] +
                    beta.values[j] * y.values[i * y.stride + j];
            }
        }
    }
}


template <typename ValueType>
inline void compute_norm2(
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>& result)
{
    for (int j = 0; j < x.num_rhs; ++j) {
        result.values[j] = gko::zero<gko::remove_complex<ValueType>>();
    }
    for (int i = 0; i < x.num_rows; ++i) {
        for (int j = 0; j < x.num_rhs; ++j) {
            result.values[j] += squared_norm(x.values[i * x.stride + j]);
        }
    }
    for (int j = 0; j < x.num_rhs; ++j) {
        result.values[j] = sqrt(result.values[j]);
    }
}


/**
 * Multiplies with a diagonal matrix represented as a dense vector.
 *
 * @param[in] diag_vec  The entries of the diagonal matrix.
 * @param[in,out] a  The dense matrix or vectors to scale.
 */
template <typename ValueType>
inline void batch_scale(
    const gko::batch_dense::BatchEntry<const ValueType>& diag_vec,
    const gko::batch_dense::BatchEntry<ValueType>& a)
{
    for (int i_row = 0; i_row < a.num_rows; i_row++) {
        const ValueType scale = diag_vec.values[i_row];
        for (int j = 0; j < a.num_rhs; j++) {
            a.values[i_row * a.stride + j] *= scale;
        }
    }
}

template <typename ValueType>
inline void batch_scale(const int nrows, const int ncols,
                        const size_type a_stride, const ValueType* const left,
                        const ValueType* const right, ValueType* const a)
{
    for (int i_row = 0; i_row < nrows; i_row++) {
        const ValueType scale = left[i_row];
        for (int j = 0; j < ncols; j++) {
            a[i_row * a_stride + j] *= scale * right[j];
        }
    }
}


/**
 * Copies the values of one multi-vector into another.
 *
 * Note that the output multi-vector should already have memory allocated
 * and stride set.
 */
template <typename ValueType>
inline void copy(const gko::batch_dense::BatchEntry<const ValueType>& in,
                 const gko::batch_dense::BatchEntry<ValueType>& out)
{
    for (int iz = 0; iz < in.num_rows * in.num_rhs; iz++) {
        const int i = iz / in.num_rhs;
        const int j = iz % in.num_rhs;
        out.values[i * out.stride + j] = in.values[i * in.stride + j];
    }
}


template <typename ValueType>
inline void compute_dot_product(
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<const ValueType>& y,
    const gko::batch_dense::BatchEntry<ValueType>& result)
{
    for (int c = 0; c < result.num_rhs; c++) {
        result.values[c] = gko::zero<ValueType>();
    }

    for (int r = 0; r < x.num_rows; r++) {
        for (int c = 0; c < x.num_rhs; c++) {
            result.values[c] +=
                conj(x.values[r * x.stride + c]) * y.values[r * y.stride + c];
        }
    }
}


template <typename ValueType>
inline void copy(
    const gko::batch_dense::BatchEntry<const ValueType>& source_entry,
    const gko::batch_dense::BatchEntry<ValueType>& destination_entry,
    const gko::uint32& converged)
{
    for (int r = 0; r < source_entry.num_rows; r++) {
        for (int c = 0; c < source_entry.num_rhs; c++) {
            const gko::uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            destination_entry.values[r * destination_entry.stride + c] =
                source_entry.values[r * source_entry.stride + c];
        }
    }
}


template <typename ValueType>
inline void add_scaled(
    const gko::batch_dense::BatchEntry<const ValueType>& alpha,
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<ValueType>& y,
    const gko::uint32& converged)
{
    if (alpha.num_rhs == 1) {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                const gko::uint32 conv = converged & (1 << j);

                if (conv) {
                    continue;
                }

                y.values[i * y.stride + j] +=
                    alpha.values[0] * x.values[i * x.stride + j];
            }
        }
    } else {
        for (int i = 0; i < x.num_rows; ++i) {
            for (int j = 0; j < x.num_rhs; ++j) {
                const gko::uint32 conv = converged & (1 << j);

                if (conv) {
                    continue;
                }


                y.values[i * y.stride + j] +=
                    alpha.values[j] * x.values[i * x.stride + j];
            }
        }
    }
}


template <typename ValueType>
inline void compute_norm2(
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>& result,
    const gko::uint32& converged)
{
    for (int j = 0; j < x.num_rhs; ++j) {
        const gko::uint32 conv = converged & (1 << j);

        if (conv) {
            continue;
        }

        result.values[j] = gko::zero<gko::remove_complex<ValueType>>();
    }
    for (int i = 0; i < x.num_rows; ++i) {
        for (int j = 0; j < x.num_rhs; ++j) {
            const gko::uint32 conv = converged & (1 << j);

            if (conv) {
                continue;
            }

            result.values[j] += squared_norm(x.values[i * x.stride + j]);
        }
    }
    for (int j = 0; j < x.num_rhs; ++j) {
        const gko::uint32 conv = converged & (1 << j);

        if (conv) {
            continue;
        }

        result.values[j] = sqrt(result.values[j]);
    }
}


template <typename ValueType>
inline void compute_dot_product(
    const gko::batch_dense::BatchEntry<const ValueType>& x,
    const gko::batch_dense::BatchEntry<const ValueType>& y,
    const gko::batch_dense::BatchEntry<ValueType>& result,
    const gko::uint32& converged)
{
    for (int c = 0; c < result.num_rhs; c++) {
        const gko::uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        result.values[c] = gko::zero<ValueType>();
    }

    for (int r = 0; r < x.num_rows; r++) {
        for (int c = 0; c < x.num_rhs; c++) {
            const gko::uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            result.values[c] +=
                conj(x.values[r * x.stride + c]) * y.values[r * y.stride + c];
        }
    }
}


template <typename ValueType>
inline void add_scaled_identity(
    const ValueType& a, const ValueType& b,
    const gko::batch_dense::BatchEntry<ValueType>& mat)
{
    for (int i = 0; i < mat.num_rows; i++) {
        for (int j = 0; j < mat.num_rhs; j++) {
            mat.values[i * mat.stride + j] *= b;
            if (i == j) {
                mat.values[i * mat.stride + i] += a;
            }
        }
    }
}
