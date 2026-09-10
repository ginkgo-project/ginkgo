// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/base/multivector.hpp>


template <typename Concrete>
class AbstractDummyVector : public gko::AbstractMultiVector {
public:
    AbstractDummyVector(std::shared_ptr<const gko::Executor> exec,
                        gko::dim<2> size = gko::dim<2>{},
                        gko::precision prec = gko::precision::fp64)
        : AbstractMultiVector(exec, size, prec)
    {}

    static std::unique_ptr<Concrete> create(
        std::shared_ptr<const gko::Executor> exec,
        gko::dim<2> size = gko::dim<2>{},
        gko::precision prec = gko::precision::fp64)
    {
        return std::make_unique<Concrete>(exec, size, prec);
    }

protected:
    Cloneable* copy_from_impl(const Cloneable* other) override
        GKO_NOT_IMPLEMENTED;
    Cloneable* copy_from_impl(std::unique_ptr<Cloneable> other) override
        GKO_NOT_IMPLEMENTED;
    Cloneable* move_from_impl(Cloneable* other) override GKO_NOT_IMPLEMENTED;
    Cloneable* move_from_impl(std::unique_ptr<Cloneable> other) override
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<Cloneable> clone_impl(
        std::shared_ptr<const gko::Executor> exec) const override
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<Cloneable> clone_impl() const override
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<Cloneable> create_default_impl()
        const override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<Cloneable> create_default_impl(
        std::shared_ptr<const gko::Executor> exec) const override
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_generic_with_same_config_impl() const override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_generic_with_type_of_impl(
        std::shared_ptr<const gko::Executor> exec,
        const gko::dim<2>& global_size, const gko::dim<2>& local_size,
        gko::size_type stride) const override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    compute_absolute_generic_impl() const override GKO_NOT_IMPLEMENTED;
    void compute_absolute_generic_impl(
        AbstractMultiVector* result) const override GKO_NOT_IMPLEMENTED;
    void compute_absolute_inplace_impl() override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    make_complex_generic_impl() const override GKO_NOT_IMPLEMENTED;
    void make_complex_generic_impl(AbstractMultiVector* result) const override
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> get_real_generic_impl()
        const override GKO_NOT_IMPLEMENTED;
    void get_real_generic_impl(AbstractMultiVector* result) const override
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector> get_imag_generic_impl()
        const override GKO_NOT_IMPLEMENTED;
    void get_imag_generic_impl(AbstractMultiVector* result) const override
        GKO_NOT_IMPLEMENTED;
    void fill_impl(gko::any_scalar value) override GKO_NOT_IMPLEMENTED;
    void scale_impl(const AbstractMultiVector* alpha) override
        GKO_NOT_IMPLEMENTED;
    void inv_scale_impl(const AbstractMultiVector* alpha) override
        GKO_NOT_IMPLEMENTED;
    void add_scaled_impl(const AbstractMultiVector* alpha,
                         const AbstractMultiVector* b) override
        GKO_NOT_IMPLEMENTED;
    void sub_scaled_impl(const AbstractMultiVector* alpha,
                         const AbstractMultiVector* b) override
        GKO_NOT_IMPLEMENTED;
    void compute_dot_impl(
        const AbstractMultiVector* b, AbstractMultiVector* result,
        gko::array<char>& tmp) const override GKO_NOT_IMPLEMENTED;
    void compute_conj_dot_impl(
        const AbstractMultiVector* b, AbstractMultiVector* result,
        gko::array<char>& tmp) const override GKO_NOT_IMPLEMENTED;
    void compute_norm2_impl(AbstractMultiVector* result, gko::array<char>& tmp)
        const override GKO_NOT_IMPLEMENTED;
    void compute_squared_norm2_impl(AbstractMultiVector* result,
                                    gko::array<char>& tmp) const override
        GKO_NOT_IMPLEMENTED;
    void compute_norm1_impl(AbstractMultiVector* result, gko::array<char>& tmp)
        const override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<const AbstractMultiVector>
    create_real_view_generic_impl() const override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_real_view_generic_impl() override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_subview_generic_impl(gko::local_span rows, gko::local_span columns)
        override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<const AbstractMultiVector>
    create_subview_generic_impl(gko::local_span rows, gko::local_span columns)
        const override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<AbstractMultiVector>
    create_subview_generic_impl(gko::local_span rows, gko::local_span columns,
                                gko::dim<2> global_size) override
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::unique_ptr<const AbstractMultiVector>
    create_subview_generic_impl(gko::local_span rows, gko::local_span columns,
                                gko::dim<2> global_size) const override
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::variant<
#if GINKGO_ENABLE_HALF
        device_view<gko::half>, device_view<std::complex<gko::half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        device_view<gko::bfloat16>, device_view<std::complex<gko::bfloat16>>,
#endif
        device_view<float>, device_view<std::complex<float>>,
        device_view<double>, device_view<std::complex<double>>>
    get_local_device_view_generic_impl() override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] std::variant<
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
        GKO_NOT_IMPLEMENTED;
    [[nodiscard]] gko::detail::temporary_conversion<AbstractMultiVector>
    as_precision_impl(gko::precision p) override GKO_NOT_IMPLEMENTED;
    [[nodiscard]] gko::detail::temporary_conversion<const AbstractMultiVector>
    as_precision_impl(gko::precision p) const override GKO_NOT_IMPLEMENTED;
};


class DummyVector : public AbstractDummyVector<DummyVector> {
public:
    using AbstractDummyVector::AbstractDummyVector;
    using AbstractDummyVector::create;
};
