// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <sstream>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/extensions/kokkos/spaces.hpp>
#include <ginkgo/extensions/kokkos/types.hpp>

#include "core/test/utils.hpp"


using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;


template <typename ValueType>
class ArrayMapper : public ::testing::Test {
protected:
    using value_type = ValueType;

    std::shared_ptr<gko::Executor> exec =
        gko::ext::kokkos::create_default_executor();
    gko::array<value_type> array = {exec, I<value_type>{1, 2, 3, 4}};
};

TYPED_TEST_SUITE(ArrayMapper, gko::test::ValueTypesBase, TypenameNameGenerator);


TYPED_TEST(ArrayMapper, CanMapDefault)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;

    auto mapped_array = gko::ext::kokkos::map_data(this->array);

    using mapped_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_array)>>;
    static_assert(
        std::is_same_v<mapped_type,
                       Kokkos::View<kokkos_value_type*, DefaultMemorySpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(mapped_array.data()),
              reinterpret_cast<std::uintptr_t>(this->array.get_data()));
    ASSERT_EQ(mapped_array.extent(0), this->array.get_size());
    ASSERT_EQ(mapped_array.size(), this->array.get_size());
    ASSERT_EQ(mapped_array.stride(0), 1);
}


TYPED_TEST(ArrayMapper, CanMapConst)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;

    auto mapped_array = gko::ext::kokkos::map_data(
        const_cast<const gko::array<value_type>&>(this->array));

    using mapped_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_array)>>;
    static_assert(std::is_same_v<
                  mapped_type,
                  Kokkos::View<const kokkos_value_type*, DefaultMemorySpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(mapped_array.data()),
              reinterpret_cast<std::uintptr_t>(this->array.get_data()));
    ASSERT_EQ(mapped_array.extent(0), this->array.get_size());
    ASSERT_EQ(mapped_array.size(), this->array.get_size());
    ASSERT_EQ(mapped_array.stride(0), 1);
}


template <typename ValueType>
class DenseMapper : public ::testing::Test {
protected:
    using value_type = ValueType;
    using mtx_type = gko::matrix::Dense<value_type>;

    std::shared_ptr<gko::Executor> exec =
        gko::ext::kokkos::create_default_executor();
    std::unique_ptr<mtx_type> mtx =
        gko::initialize<mtx_type>({1, 2, 3, 4}, exec);
};

TYPED_TEST_SUITE(DenseMapper, gko::test::ValueTypesBase, TypenameNameGenerator);


TYPED_TEST(DenseMapper, CanMapDefault)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;

    auto mapped_mtx = gko::ext::kokkos::map_data(this->mtx);

    using mapped_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_mtx)>>;
    static_assert(
        std::is_same_v<mapped_type,
                       Kokkos::View<kokkos_value_type**, Kokkos::LayoutStride,
                                    DefaultMemorySpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(mapped_mtx.data()),
              reinterpret_cast<std::uintptr_t>(this->mtx->get_values()));
    ASSERT_EQ(mapped_mtx.extent(0), this->mtx->get_size()[0]);
    ASSERT_EQ(mapped_mtx.extent(1), this->mtx->get_size()[1]);
    ASSERT_EQ(mapped_mtx.size(),
              this->mtx->get_size()[0] * this->mtx->get_size()[1]);
    ASSERT_EQ(mapped_mtx.span(), this->mtx->get_num_stored_elements());
    ASSERT_EQ(mapped_mtx.stride(0), this->mtx->get_stride());
    ASSERT_EQ(mapped_mtx.stride(1), 1);
}


TYPED_TEST(DenseMapper, CanMapConst)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;

    auto mapped_mtx = gko::ext::kokkos::map_data(
        const_cast<const gko::matrix::Dense<value_type>*>(this->mtx.get()));

    using mapped_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_mtx)>>;
    static_assert(
        std::is_same_v<mapped_type,
                       Kokkos::View<const kokkos_value_type**,
                                    Kokkos::LayoutStride, DefaultMemorySpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(mapped_mtx.data()),
              reinterpret_cast<std::uintptr_t>(this->mtx->get_values()));
    ASSERT_EQ(mapped_mtx.extent(0), this->mtx->get_size()[0]);
    ASSERT_EQ(mapped_mtx.extent(1), this->mtx->get_size()[1]);
    ASSERT_EQ(mapped_mtx.size(),
              this->mtx->get_size()[0] * this->mtx->get_size()[1]);
    ASSERT_EQ(mapped_mtx.span(), this->mtx->get_num_stored_elements());
    ASSERT_EQ(mapped_mtx.stride(0), this->mtx->get_stride());
    ASSERT_EQ(mapped_mtx.stride(1), 1);
}


TYPED_TEST(DenseMapper, CanMapStrided)
{
    using mtx_type = typename TestFixture::mtx_type;
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;
    std::unique_ptr<mtx_type> mtx = mtx_type ::create(
        this->exec, gko::dim<2>{2, 2},
        gko::array<value_type>{this->exec, {1, 2, -1, 3, 4, -10}}, 3);

    auto mapped_mtx = gko::ext::kokkos::map_data(mtx);

    using mapped_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_mtx)>>;
    static_assert(
        std::is_same_v<mapped_type,
                       Kokkos::View<kokkos_value_type**, Kokkos::LayoutStride,
                                    DefaultMemorySpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(mapped_mtx.data()),
              reinterpret_cast<std::uintptr_t>(mtx->get_values()));
    ASSERT_EQ(mapped_mtx.extent(0), mtx->get_size()[0]);
    ASSERT_EQ(mapped_mtx.extent(1), mtx->get_size()[1]);
    ASSERT_EQ(mapped_mtx.size(), mtx->get_size()[0] * mtx->get_size()[1]);
    ASSERT_EQ(mapped_mtx.span(), mtx->get_num_stored_elements());
    ASSERT_EQ(mapped_mtx.stride(0), mtx->get_stride());
    ASSERT_EQ(mapped_mtx.stride(1), 1);
}
