// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/fft.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/fft_kernels.hpp"


namespace gko {
namespace matrix {
namespace fft {
namespace {


GKO_REGISTER_OPERATION(fft, fft::fft);
GKO_REGISTER_OPERATION(fft2, fft::fft2);
GKO_REGISTER_OPERATION(fft3, fft::fft3);


template <typename ValueType, typename IndexType>
void write_impl_1d(int64 size, bool inverse,
                   matrix_data<ValueType, IndexType>& data)
{
    auto usize = static_cast<size_type>(size);
    data.size = dim<2>{usize, usize};
    data.nonzeros.assign(usize * usize, {0, 0, zero<ValueType>()});
    int sign = inverse ? 1 : -1;
    for (int64 row = 0; row < size; row++) {
        for (int64 col = 0; col < size; col++) {
            data.nonzeros[row * size + col] = {
                static_cast<IndexType>(row), static_cast<IndexType>(col),
                gko::unit_root<ValueType>(size, sign * ((row * col) % size))};
        }
    }
}


template <typename ValueType, typename IndexType>
void write_impl_2d(int64 size1, int64 size2, bool inverse,
                   matrix_data<ValueType, IndexType>& data)
{
    const auto size = size1 * size2;
    const auto usize = static_cast<size_type>(size);
    data.size = dim<2>{usize, usize};
    data.nonzeros.assign(usize * usize, {0, 0, zero<ValueType>()});
    int sign = inverse ? 1 : -1;
    for (int64 i1 = 0; i1 < size1; i1++) {
        for (int64 i2 = 0; i2 < size2; i2++) {
            for (int64 j1 = 0; j1 < size1; j1++) {
                for (int64 j2 = 0; j2 < size2; j2++) {
                    auto row = i1 * size2 + i2;
                    auto col = j1 * size2 + j2;
                    data.nonzeros[row * size + col] = {
                        static_cast<IndexType>(row),
                        static_cast<IndexType>(col),
                        gko::unit_root<ValueType>(size1,
                                                  sign * ((i1 * j1) % size1)) *
                            gko::unit_root<ValueType>(
                                size2, sign * ((i2 * j2) % size2))};
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void write_impl_3d(int64 size1, int64 size2, int64 size3, bool inverse,
                   matrix_data<ValueType, IndexType>& data)
{
    const auto size = size1 * size2 * size3;
    const auto usize = static_cast<size_type>(size);
    data.size = dim<2>{usize, usize};
    data.nonzeros.assign(usize * usize, {0, 0, zero<ValueType>()});
    int sign = inverse ? 1 : -1;
    for (int64 i1 = 0; i1 < size1; i1++) {
        for (int64 i2 = 0; i2 < size2; i2++) {
            for (int64 i3 = 0; i3 < size3; i3++) {
                for (int64 j1 = 0; j1 < size1; j1++) {
                    for (int64 j2 = 0; j2 < size2; j2++) {
                        for (int64 j3 = 0; j3 < size3; j3++) {
                            auto row = i1 * size2 * size3 + i2 * size3 + i3;
                            auto col = j1 * size2 * size3 + j2 * size3 + j3;
                            data.nonzeros[row * size + col] = {
                                static_cast<IndexType>(row),
                                static_cast<IndexType>(col),
                                gko::unit_root<ValueType>(
                                    size1, sign * ((i1 * j1) % size1)) *
                                    gko::unit_root<ValueType>(
                                        size2, sign * ((i2 * j2) % size2)) *
                                    gko::unit_root<ValueType>(
                                        size3, sign * ((i3 * j3) % size3))};
                        }
                    }
                }
            }
        }
    }
}


}  // namespace
}  // namespace fft


std::unique_ptr<LinOp> Fft::transpose() const
{
    return Fft::create(this->get_executor(), this->get_size()[0], inverse_);
}


std::unique_ptr<LinOp> Fft::conj_transpose() const
{
    return Fft::create(this->get_executor(), this->get_size()[0], !inverse_);
}


void Fft::write(matrix_data<std::complex<float>, int32>& data) const
{
    fft::write_impl_1d(get_size()[0], is_inverse(), data);
}


void Fft::write(matrix_data<std::complex<float>, int64>& data) const
{
    fft::write_impl_1d(get_size()[0], is_inverse(), data);
}


void Fft::write(matrix_data<std::complex<double>, int32>& data) const
{
    fft::write_impl_1d(get_size()[0], is_inverse(), data);
}


void Fft::write(matrix_data<std::complex<double>, int64>& data) const
{
    fft::write_impl_1d(get_size()[0], is_inverse(), data);
}


dim<1> Fft::get_fft_size() const { return dim<1>{this->get_size()[0]}; }


bool Fft::is_inverse() const { return inverse_; }


void Fft::apply_impl(const LinOp* b, LinOp* x) const
{
    if (auto float_b = dynamic_cast<const Dense<std::complex<float>>*>(b)) {
        auto dense_x = as<Dense<std::complex<float>>>(x);
        get_executor()->run(fft::make_fft(float_b, dense_x, inverse_, buffer_));
    } else {
        auto dense_b = as<Dense<std::complex<double>>>(b);
        auto dense_x = as<Dense<std::complex<double>>>(x);
        get_executor()->run(fft::make_fft(dense_b, dense_x, inverse_, buffer_));
    }
}


void Fft::apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                     LinOp* x) const
{
    if (auto float_x = dynamic_cast<Dense<std::complex<float>>*>(x)) {
        auto clone_x = x->clone();
        this->apply_impl(b, clone_x.get());
        float_x->scale(beta);
        float_x->add_scaled(alpha, clone_x);
    } else {
        auto dense_x = as<Dense<std::complex<double>>>(x);
        auto clone_x = x->clone();
        this->apply_impl(b, clone_x.get());
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, clone_x);
    }
}


std::unique_ptr<LinOp> Fft2::transpose() const
{
    return Fft2::create(this->get_executor(), fft_size_[0], fft_size_[1],
                        inverse_);
}


std::unique_ptr<LinOp> Fft2::conj_transpose() const
{
    return Fft2::create(this->get_executor(), fft_size_[0], fft_size_[1],
                        !inverse_);
}


void Fft2::write(matrix_data<std::complex<float>, int32>& data) const
{
    fft::write_impl_2d(fft_size_[0], fft_size_[1], is_inverse(), data);
}


void Fft2::write(matrix_data<std::complex<float>, int64>& data) const
{
    fft::write_impl_2d(fft_size_[0], fft_size_[1], is_inverse(), data);
}


void Fft2::write(matrix_data<std::complex<double>, int32>& data) const
{
    fft::write_impl_2d(fft_size_[0], fft_size_[1], is_inverse(), data);
}


void Fft2::write(matrix_data<std::complex<double>, int64>& data) const
{
    fft::write_impl_2d(fft_size_[0], fft_size_[1], is_inverse(), data);
}


dim<2> Fft2::get_fft_size() const { return fft_size_; }


bool Fft2::is_inverse() const { return inverse_; }


void Fft2::apply_impl(const LinOp* b, LinOp* x) const
{
    if (auto float_b = dynamic_cast<const Dense<std::complex<float>>*>(b)) {
        auto dense_x = as<Dense<std::complex<float>>>(x);
        get_executor()->run(fft::make_fft2(float_b, dense_x, fft_size_[0],
                                           fft_size_[1], inverse_, buffer_));
    } else {
        auto dense_b = as<Dense<std::complex<double>>>(b);
        auto dense_x = as<Dense<std::complex<double>>>(x);
        get_executor()->run(fft::make_fft2(dense_b, dense_x, fft_size_[0],
                                           fft_size_[1], inverse_, buffer_));
    }
}


void Fft2::apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                      LinOp* x) const
{
    if (auto float_x = dynamic_cast<Dense<std::complex<float>>*>(x)) {
        auto clone_x = x->clone();
        this->apply_impl(b, clone_x.get());
        float_x->scale(beta);
        float_x->add_scaled(alpha, clone_x);
    } else {
        auto dense_x = as<Dense<std::complex<double>>>(x);
        auto clone_x = x->clone();
        this->apply_impl(b, clone_x.get());
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, clone_x);
    }
}


std::unique_ptr<LinOp> Fft3::transpose() const
{
    return Fft3::create(this->get_executor(), fft_size_[0], fft_size_[1],
                        fft_size_[2], inverse_);
}


std::unique_ptr<LinOp> Fft3::conj_transpose() const
{
    return Fft3::create(this->get_executor(), fft_size_[0], fft_size_[1],
                        fft_size_[2], !inverse_);
}


void Fft3::write(matrix_data<std::complex<float>, int32>& data) const
{
    fft::write_impl_3d(fft_size_[0], fft_size_[1], fft_size_[2], is_inverse(),
                       data);
}


void Fft3::write(matrix_data<std::complex<float>, int64>& data) const
{
    fft::write_impl_3d(fft_size_[0], fft_size_[1], fft_size_[2], is_inverse(),
                       data);
}


void Fft3::write(matrix_data<std::complex<double>, int32>& data) const
{
    fft::write_impl_3d(fft_size_[0], fft_size_[1], fft_size_[2], is_inverse(),
                       data);
}


void Fft3::write(matrix_data<std::complex<double>, int64>& data) const
{
    fft::write_impl_3d(fft_size_[0], fft_size_[1], fft_size_[2], is_inverse(),
                       data);
}


dim<3> Fft3::get_fft_size() const { return fft_size_; }


bool Fft3::is_inverse() const { return inverse_; }


void Fft3::apply_impl(const LinOp* b, LinOp* x) const
{
    if (auto float_b = dynamic_cast<const Dense<std::complex<float>>*>(b)) {
        auto dense_x = as<Dense<std::complex<float>>>(x);
        get_executor()->run(fft::make_fft3(float_b, dense_x, fft_size_[0],
                                           fft_size_[1], fft_size_[2], inverse_,
                                           buffer_));
    } else {
        auto dense_b = as<Dense<std::complex<double>>>(b);
        auto dense_x = as<Dense<std::complex<double>>>(x);
        get_executor()->run(fft::make_fft3(dense_b, dense_x, fft_size_[0],
                                           fft_size_[1], fft_size_[2], inverse_,
                                           buffer_));
    }
}


void Fft3::apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                      LinOp* x) const
{
    if (auto float_x = dynamic_cast<Dense<std::complex<float>>*>(x)) {
        auto clone_x = x->clone();
        this->apply_impl(b, clone_x.get());
        float_x->scale(beta);
        float_x->add_scaled(alpha, clone_x);
    } else {
        auto dense_x = as<Dense<std::complex<double>>>(x);
        auto clone_x = x->clone();
        this->apply_impl(b, clone_x.get());
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, clone_x);
    }
}


}  // namespace matrix
}  // namespace gko
