// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <set>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/dummy_vector.hpp"


class ScalarVector : public DummyVector, public gko::matrix::MultiVector<> {
public:
    ScalarVector(std::shared_ptr<const gko::Executor> exec,
                 gko::dim<2> size = {})
        : DummyVector(exec, size), MultiVector<>(exec, size)
    {}
};


class TrackingVector : public AbstractDummyVector<TrackingVector> {
public:
    enum class function {
        create_with_same_config,
        create_with_type_of,
        compute_absolute,
        compute_absolute_output,
        compute_absolute_inplace,
        make_complex,
        make_complex_output,
        get_real,
        get_real_output,
        get_imag,
        get_imag_output,
        fill,
        scale,
        inv_scale,
        add_scaled,
        sub_scaled,
        dot_with_tmp,
        conj_dot_with_tmp,
        norm2_with_tmp,
        squared_norm2_with_tmp,
        norm1_with_tmp,
        real_view,
        const_real_view,
        subview,
        const_subview,
        subview_with_global_size,
        const_subview_with_global_size,
        local_device_view,
        const_local_device_view
    };

    struct expected_result {
        expected_result(function func, int count = 1) : func(func), count(count)
        {}

        function func;
        int count;
    };

    using AbstractDummyVector::AbstractDummyVector;

    bool was_called(expected_result r)
    {
        return calls.count(r.func) == r.count;
    }

protected:
    void mark(function impl) const { calls.insert(impl); }

    std::unique_ptr<AbstractMultiVector> create_generic_with_same_config_impl()
        const override
    {
        mark(function::create_with_same_config);
        return {};
    }

    std::unique_ptr<AbstractMultiVector> create_generic_with_type_of_impl(
        std::shared_ptr<const gko::Executor>, const gko::dim<2>&,
        const gko::dim<2>&, gko::size_type) const override
    {
        mark(function::create_with_type_of);
        return {};
    }

    std::unique_ptr<AbstractMultiVector> compute_absolute_generic_impl()
        const override
    {
        mark(function::compute_absolute);
        return {};
    }

    void compute_absolute_generic_impl(AbstractMultiVector*) const override
    {
        mark(function::compute_absolute_output);
    }

    void compute_absolute_inplace_impl() override
    {
        mark(function::compute_absolute_inplace);
    }

    std::unique_ptr<AbstractMultiVector> make_complex_generic_impl()
        const override
    {
        mark(function::make_complex);
        return {};
    }

    void make_complex_generic_impl(AbstractMultiVector*) const override
    {
        mark(function::make_complex_output);
    }

    std::unique_ptr<AbstractMultiVector> get_real_generic_impl() const override
    {
        mark(function::get_real);
        return {};
    }

    void get_real_generic_impl(AbstractMultiVector*) const override
    {
        mark(function::get_real_output);
    }

    std::unique_ptr<AbstractMultiVector> get_imag_generic_impl() const override
    {
        mark(function::get_imag);
        return {};
    }

    void get_imag_generic_impl(AbstractMultiVector*) const override
    {
        mark(function::get_imag_output);
    }

    void fill_impl(gko::any_scalar) override { mark(function::fill); }

    void scale_impl(const AbstractMultiVector*) override
    {
        mark(function::scale);
    }

    void inv_scale_impl(const AbstractMultiVector*) override
    {
        mark(function::inv_scale);
    }

    void add_scaled_impl(const AbstractMultiVector*,
                         const AbstractMultiVector*) override
    {
        mark(function::add_scaled);
    }

    void sub_scaled_impl(const AbstractMultiVector*,
                         const AbstractMultiVector*) override
    {
        mark(function::sub_scaled);
    }

    void compute_dot_impl(const AbstractMultiVector*, AbstractMultiVector*,
                          gko::array<char>&) const override
    {
        mark(function::dot_with_tmp);
    }

    void compute_conj_dot_impl(const AbstractMultiVector*, AbstractMultiVector*,
                               gko::array<char>&) const override
    {
        mark(function::conj_dot_with_tmp);
    }

    void compute_norm2_impl(AbstractMultiVector*,
                            gko::array<char>&) const override
    {
        mark(function::norm2_with_tmp);
    }

    void compute_squared_norm2_impl(AbstractMultiVector*,
                                    gko::array<char>&) const override
    {
        mark(function::squared_norm2_with_tmp);
    }

    void compute_norm1_impl(AbstractMultiVector*,
                            gko::array<char>&) const override
    {
        mark(function::norm1_with_tmp);
    }

    std::unique_ptr<const AbstractMultiVector> create_real_view_generic_impl()
        const override
    {
        mark(function::const_real_view);
        return {};
    }

    std::unique_ptr<AbstractMultiVector> create_real_view_generic_impl()
        override
    {
        mark(function::real_view);
        return {};
    }

    std::unique_ptr<AbstractMultiVector> create_subview_generic_impl(
        gko::local_span, gko::local_span) override
    {
        mark(function::subview);
        return {};
    }

    std::unique_ptr<const AbstractMultiVector> create_subview_generic_impl(
        gko::local_span, gko::local_span) const override
    {
        mark(function::const_subview);
        return {};
    }

    std::unique_ptr<AbstractMultiVector> create_subview_generic_impl(
        gko::local_span, gko::local_span, gko::dim<2>) override
    {
        mark(function::subview_with_global_size);
        return {};
    }

    std::unique_ptr<const AbstractMultiVector> create_subview_generic_impl(
        gko::local_span, gko::local_span, gko::dim<2>) const override
    {
        mark(function::const_subview_with_global_size);
        return {};
    }

    std::variant<
#if GINKGO_ENABLE_HALF
        device_view<gko::half>, device_view<std::complex<gko::half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        device_view<gko::bfloat16>, device_view<std::complex<gko::bfloat16>>,
#endif
        device_view<float>, device_view<std::complex<float>>,
        device_view<double>, device_view<std::complex<double>>>
    get_local_device_view_generic_impl() override
    {
        mark(function::local_device_view);
        return device_view<double>{{}, 0, nullptr};
    }

    std::variant<
#if GINKGO_ENABLE_HALF
        device_view<const gko::half>,
        device_view<const std::complex<gko::half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        device_view<const gko::bfloat16>,
        device_view<const std::complex<gko::bfloat16>>,
#endif
        device_view<const float>, device_view<const std::complex<float>>,
        device_view<const double>, device_view<const std::complex<double>>>
    get_const_local_device_view_generic_impl() const override
    {
        mark(function::const_local_device_view);
        return device_view<const double>{{}, 0, nullptr};
    }

private:
    mutable std::multiset<function> calls;
};


class AbstractMultiVector : public ::testing::Test {
protected:
    AbstractMultiVector()
        : exec(gko::ReferenceExecutor::create()),
          vector(TrackingVector::create(exec, {2, 3}, gko::precision::fp64)),
          other(DummyVector::create(exec, {2, 3}, gko::precision::fp64)),
          result(DummyVector::create(exec, {1, 3}, gko::precision::fp64)),
          alpha(std::make_unique<ScalarVector>(exec, gko::dim<2>{1, 1})),
          tmp(exec)
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<TrackingVector> vector;
    std::unique_ptr<DummyVector> other;
    std::unique_ptr<DummyVector> result;
    std::unique_ptr<ScalarVector> alpha;
    gko::array<char> tmp;
};


TEST_F(AbstractMultiVector, DispatchToImplementations)
{
    auto other_complex = DummyVector::create(
        exec, other->get_size(), gko::as_complex(other->get_precision()));

    (void)gko::AbstractMultiVector::create_with_config_of(vector);
    (void)gko::AbstractMultiVector::create_with_type_of(vector, exec);
    (void)gko::AbstractMultiVector::create_with_type_of(
        vector, exec, gko::dim<2>{2, 3}, gko::dim<2>{2, 3});
    (void)gko::AbstractMultiVector::create_with_type_of(
        vector, exec, gko::dim<2>{2, 3}, gko::dim<2>{2, 3}, 3);
    (void)vector->compute_absolute();
    vector->compute_absolute(other);
    vector->compute_absolute_inplace();
    (void)vector->make_complex();
    vector->make_complex(other_complex);
    (void)vector->get_real();
    vector->get_real(other);
    (void)vector->get_imag();
    vector->get_imag(other);
    vector->fill(1.0);

    auto alpha = std::make_shared<ScalarVector>(exec, gko::dim<2>{1, 1});
    vector->scale(alpha);
    vector->inv_scale(alpha);
    vector->add_scaled(alpha, other);
    vector->sub_scaled(alpha, other);
    vector->compute_dot(other, result);
    vector->compute_dot(other, result, tmp);
    vector->compute_conj_dot(other, result);
    vector->compute_conj_dot(other, result, tmp);
    vector->compute_norm2(result);
    vector->compute_norm2(result, tmp);
    vector->compute_squared_norm2(result);
    vector->compute_squared_norm2(result, tmp);
    vector->compute_norm1(result);
    vector->compute_norm1(result, tmp);
    (void)vector->create_real_view();
    (void)vector->create_subview(gko::span{0, 1}, gko::span{0, 1});
    (void)vector->create_subview(gko::span{0, 1}, gko::span{0, 1},
                                 gko::dim<2>{2, 3});
    auto const_vector =
        static_cast<const gko::AbstractMultiVector*>(vector.get());
    (void)const_vector->create_real_view();
    (void)const_vector->create_subview(gko::span{0, 1}, gko::span{0, 1});
    (void)const_vector->create_subview(gko::span{0, 1}, gko::span{0, 1},
                                       gko::dim<2>{2, 3});
    (void)vector->get_local_device_view<double>();
    (void)const_vector->get_const_local_device_view<double>();

    using F = TrackingVector::function;
    for (auto impl :
         I<TrackingVector::expected_result>{F::create_with_same_config,
                                            {F::create_with_type_of, 3},
                                            F::compute_absolute,
                                            F::compute_absolute_output,
                                            F::compute_absolute_inplace,
                                            F::make_complex,
                                            F::make_complex_output,
                                            F::get_real,
                                            F::get_real_output,
                                            F::get_imag,
                                            F::get_imag_output,
                                            F::fill,
                                            F::scale,
                                            F::inv_scale,
                                            F::add_scaled,
                                            F::sub_scaled,
                                            {F::dot_with_tmp, 2},
                                            {F::conj_dot_with_tmp, 2},
                                            {F::norm2_with_tmp, 2},
                                            {F::squared_norm2_with_tmp, 2},
                                            {F::norm1_with_tmp, 2},
                                            F::real_view,
                                            F::const_real_view,
                                            F::subview,
                                            F::const_subview,
                                            F::subview_with_global_size,
                                            F::const_subview_with_global_size,
                                            F::local_device_view,
                                            F::const_local_device_view}) {
        EXPECT_TRUE(vector->was_called(impl));
    }
}


TEST_F(AbstractMultiVector, CreateWithTypeOfThrows)
{
    auto wrong_vector = DummyVector::create(exec, {4, 3});

    EXPECT_THROW(auto v = gko::AbstractMultiVector::create_with_type_of(
                     vector, exec, gko::dim<2>{2, 3}, gko::dim<2>{2, 2}),
                 gko::DimensionMismatch);
}


TEST_F(AbstractMultiVector, ComputeAbsoluteThrows)
{
    auto wrong_vector = DummyVector::create(exec, {4, 3});
    auto wrong_precision =
        DummyVector::create(exec, vector->get_size(), gko::precision::fp32);

    EXPECT_THROW(vector->compute_absolute(wrong_vector),
                 gko::DimensionMismatch);
    EXPECT_THROW(vector->compute_absolute(wrong_precision),
                 gko::PrecisionError);
}


TEST_F(AbstractMultiVector, MakeComplexThrows)
{
    auto wrong_vector = DummyVector::create(exec, {4, 3});
    auto wrong_precision =
        DummyVector::create(exec, vector->get_size(), gko::precision::fp32);

    EXPECT_THROW(vector->make_complex(wrong_vector), gko::DimensionMismatch);
    EXPECT_THROW(vector->make_complex(wrong_precision), gko::PrecisionError);
}


TEST_F(AbstractMultiVector, GetRealThrows)
{
    auto wrong_vector = DummyVector::create(exec, {4, 3});
    auto wrong_precision =
        DummyVector::create(exec, vector->get_size(), gko::precision::fp32);

    EXPECT_THROW(vector->get_real(wrong_vector), gko::DimensionMismatch);
    EXPECT_THROW(vector->get_real(wrong_precision), gko::PrecisionError);
}


TEST_F(AbstractMultiVector, GetImagThrows)
{
    auto wrong_vector = DummyVector::create(exec, {4, 3});
    auto wrong_precision =
        DummyVector::create(exec, vector->get_size(), gko::precision::fp32);

    EXPECT_THROW(vector->get_imag(wrong_vector), gko::DimensionMismatch);
    EXPECT_THROW(vector->get_imag(wrong_precision), gko::PrecisionError);
}


TEST_F(AbstractMultiVector, ScaleThrows)
{
    auto wrong_cols = std::make_shared<ScalarVector>(exec, gko::dim<2>{1, 4});
    auto wrong_rows = std::make_shared<ScalarVector>(exec, gko::dim<2>{2, 1});
    auto wrong_alpha = DummyVector::create(exec, gko::dim<2>{2, 1});

    EXPECT_THROW(vector->scale(wrong_cols), gko::DimensionMismatch);
    EXPECT_THROW(vector->scale(wrong_rows), gko::DimensionMismatch);
    EXPECT_THROW(vector->scale(wrong_alpha), gko::NotSupported);
}


TEST_F(AbstractMultiVector, InvScaleThrows)
{
    auto wrong_cols = std::make_shared<ScalarVector>(exec, gko::dim<2>{1, 4});
    auto wrong_rows = std::make_shared<ScalarVector>(exec, gko::dim<2>{2, 1});
    auto wrong_alpha = DummyVector::create(exec, gko::dim<2>{2, 1});

    EXPECT_THROW(vector->inv_scale(wrong_cols), gko::DimensionMismatch);
    EXPECT_THROW(vector->inv_scale(wrong_rows), gko::DimensionMismatch);
    EXPECT_THROW(vector->inv_scale(wrong_alpha), gko::NotSupported);
}


TEST_F(AbstractMultiVector, AddScaledThrows)
{
    auto wrong_cols = std::make_shared<ScalarVector>(exec, gko::dim<2>{1, 4});
    auto wrong_rows = std::make_shared<ScalarVector>(exec, gko::dim<2>{2, 1});
    auto wrong_alpha = DummyVector::create(exec, gko::dim<2>{2, 1});
    auto wrong_other = DummyVector::create(exec, gko::dim<2>{3, 2});

    EXPECT_THROW(vector->add_scaled(wrong_cols, other), gko::DimensionMismatch);
    EXPECT_THROW(vector->add_scaled(wrong_rows, other), gko::DimensionMismatch);
    EXPECT_THROW(vector->add_scaled(wrong_alpha, other), gko::NotSupported);
    EXPECT_THROW(vector->add_scaled(alpha, wrong_other),
                 gko::DimensionMismatch);
}


TEST_F(AbstractMultiVector, SubScaledThrows)
{
    auto wrong_cols = std::make_shared<ScalarVector>(exec, gko::dim<2>{1, 4});
    auto wrong_rows = std::make_shared<ScalarVector>(exec, gko::dim<2>{2, 1});
    auto wrong_alpha = DummyVector::create(exec, gko::dim<2>{2, 1});
    auto wrong_other = DummyVector::create(exec, gko::dim<2>{3, 2});

    EXPECT_THROW(vector->sub_scaled(wrong_cols, other), gko::DimensionMismatch);
    EXPECT_THROW(vector->sub_scaled(wrong_rows, other), gko::DimensionMismatch);
    EXPECT_THROW(vector->sub_scaled(wrong_alpha, other), gko::NotSupported);
    EXPECT_THROW(vector->sub_scaled(alpha, wrong_other),
                 gko::DimensionMismatch);
}


TEST_F(AbstractMultiVector, ComputeDotThrows)
{
    auto wrong_vector = DummyVector::create(exec, {4, 3});
    auto wrong_norm = DummyVector::create(exec, {1, 2});

    EXPECT_THROW(vector->compute_dot(wrong_vector, result),
                 gko::DimensionMismatch);
    EXPECT_THROW(vector->compute_dot(other, wrong_norm),
                 gko::DimensionMismatch);
}


TEST_F(AbstractMultiVector, ComputeConjDotThrows)
{
    auto wrong_vector = DummyVector::create(exec, {4, 3});
    auto wrong_norm = DummyVector::create(exec, {1, 2});

    EXPECT_THROW(vector->compute_conj_dot(wrong_vector, result),
                 gko::DimensionMismatch);
    EXPECT_THROW(vector->compute_conj_dot(other, wrong_norm),
                 gko::DimensionMismatch);
}


TEST_F(AbstractMultiVector, ComputeNorm2Throws)
{
    auto wrong_norm = DummyVector::create(exec, {1, 2});

    EXPECT_THROW(vector->compute_norm2(wrong_norm), gko::DimensionMismatch);
    EXPECT_THROW(vector->compute_norm2(wrong_norm, tmp),
                 gko::DimensionMismatch);
}


TEST_F(AbstractMultiVector, ComputeSquaredNorm2Throws)
{
    auto wrong_norm = DummyVector::create(exec, {1, 2});

    EXPECT_THROW(vector->compute_squared_norm2(wrong_norm),
                 gko::DimensionMismatch);
    EXPECT_THROW(vector->compute_squared_norm2(wrong_norm, tmp),
                 gko::DimensionMismatch);
}


TEST_F(AbstractMultiVector, ComputeNorm1Throws)
{
    auto wrong_norm = DummyVector::create(exec, {1, 2});

    EXPECT_THROW(vector->compute_norm1(wrong_norm), gko::DimensionMismatch);
    EXPECT_THROW(vector->compute_norm1(wrong_norm, tmp),
                 gko::DimensionMismatch);
}
