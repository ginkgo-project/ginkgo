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

#include <ginkgo/extensions/kokkos/types.hpp>


#include <cstring>
#include <sstream>


#include <gtest/gtest.h>


#include <ginkgo/core/base/native_type.hpp>


#include "core/test/utils.hpp"


using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

template <typename ValueType>
class ArrayMapper : public ::testing::Test {
protected:
    using value_type = ValueType;

    std::shared_ptr<gko::ReferenceExecutor> ref =
        gko::ReferenceExecutor::create();
    gko::array<value_type> array = {ref, I<value_type>{1, 2, 3, 4}};
};

TYPED_TEST_SUITE(ArrayMapper, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(ArrayMapper, CanMapDefault)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::value_type<value_type>::type;

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
        typename gko::ext::kokkos::value_type<value_type>::type;

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

    std::shared_ptr<gko::ReferenceExecutor> ref =
        gko::ReferenceExecutor::create();
    std::unique_ptr<mtx_type> mtx =
        gko::initialize<mtx_type>({1, 2, 3, 4}, ref);
};

TYPED_TEST_SUITE(DenseMapper, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(DenseMapper, CanMapDefault)
{
    using value_type = typename TestFixture::value_type;
    using kokkos_value_type =
        typename gko::ext::kokkos::value_type<value_type>::type;

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
        typename gko::ext::kokkos::value_type<value_type>::type;

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
