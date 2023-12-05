// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstring>
#include <sstream>


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

TYPED_TEST_SUITE(ArrayMapper, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(ArrayMapper, CanMapDefault)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;

    auto mapped_array = gko::ext::kokkos::map_data(this->array);

    using array_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_array)>>;
    static_assert(
        std::is_same_v<array_type,
                       Kokkos::View<kokkos_value_type*, DefaultMemorySpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
}

TYPED_TEST(ArrayMapper, CanMapConst)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;

    auto mapped_array = gko::ext::kokkos::map_data(
        const_cast<const gko::array<value_type>&>(this->array));

    using array_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_array)>>;
    static_assert(std::is_same_v<
                  array_type,
                  Kokkos::View<const kokkos_value_type*, DefaultMemorySpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
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

TYPED_TEST_SUITE(DenseMapper, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(DenseMapper, CanMapDefault)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;

    auto mapped_mtx = gko::ext::kokkos::map_data(this->mtx);

    using mtx_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_mtx)>>;
    static_assert(
        std::is_same_v<mtx_type,
                       Kokkos::View<kokkos_value_type**, Kokkos::LayoutStride,
                                    DefaultMemorySpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
}

TYPED_TEST(DenseMapper, CanMapConst)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::detail::value_type<value_type>::type;

    auto mapped_mtx = gko::ext::kokkos::map_data(
        const_cast<const gko::matrix::Dense<value_type>*>(this->mtx.get()));

    using mtx_type =
        std::remove_cv_t<std::remove_reference_t<decltype(mapped_mtx)>>;
    static_assert(
        std::is_same_v<mtx_type,
                       Kokkos::View<const kokkos_value_type**,
                                    Kokkos::LayoutStride, DefaultMemorySpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>);
}
