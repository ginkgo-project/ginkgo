// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/*@GKO_PREPROCESSOR_FILENAME_HELPER@*/

#include <type_traits>

#include <gtest/gtest.h>

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/array_access.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


template <typename ValueType>
class MultiVectorView : public CommonTestFixture {
public:
    using value_type = ValueType;
    using view_type = gko::matrix::view::dense<value_type>;
};

TYPED_TEST_SUITE(MultiVectorView, gko::test::ValueTypes, TypenameNameGenerator);


template <typename ValueType>
void assert_dense_view(std::shared_ptr<const gko::EXEC_TYPE> exec)
{
    gko::array<bool> correct{exec, {false}};
    gko::array<ValueType> values{exec, 6};
    values.fill(gko::one<ValueType>());
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto values, auto correct) {
            using device_type = std::decay_t<decltype(values[0])>;
            gko::matrix::view::dense<device_type> view{gko::dim<2>{2, 2}, 3,
                                                       values};
            if (view.size == gko::dim<2>(2, 2) && view.stride == 3 &&
                view.values == values && &view(0, 0) == &values[0] &&
                &view(1, 0) == &values[3] && &view(1, 1) == &values[4] &&
                view(1, 1) == gko::one(view(1, 1))) {
                *correct = true;
            }
        },
        1, values, correct);
    ASSERT_TRUE(get_element(correct, 0));
}

TYPED_TEST(MultiVectorView, WorksOnDevice)
{
    assert_dense_view<TypeParam>(this->exec);
}


template <typename ValueIndexType>
class CooView : public CommonTestFixture {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(CooView, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


template <typename ValueType, typename IndexType>
void assert_coo_view(std::shared_ptr<const gko::EXEC_TYPE> exec)
{
    gko::array<bool> correct{exec, {false}};
    gko::array<ValueType> values{exec, 3};
    gko::array<IndexType> row_idxs{exec, 3};
    gko::array<IndexType> col_idxs{exec, 3};
    values.fill(gko::one<ValueType>());
    row_idxs.fill(IndexType{0});
    col_idxs.fill(IndexType{1});
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto values, auto row_idxs, auto col_idxs,
                      auto correct) {
            using vt = std::decay_t<decltype(values[0])>;
            using it = std::decay_t<decltype(row_idxs[0])>;
            gko::matrix::view::coo<vt, it> view{gko::dim<2>{3, 3}, 3, values,
                                                row_idxs, col_idxs};
            if (view.size == gko::dim<2>(3, 3) &&
                view.num_stored_elements == 3 && view.values == values &&
                view.row_idxs == row_idxs && view.col_idxs == col_idxs) {
                *correct = true;
            }
        },
        1, values, row_idxs, col_idxs, correct);
    ASSERT_TRUE(get_element(correct, 0));
}

TYPED_TEST(CooView, WorksOnDevice)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    assert_coo_view<value_type, index_type>(this->exec);
}


template <typename ValueIndexType>
class CsrView : public CommonTestFixture {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(CsrView, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


template <typename ValueType, typename IndexType>
void assert_csr_view(std::shared_ptr<const gko::EXEC_TYPE> exec)
{
    gko::array<bool> correct{exec, {false}};
    gko::array<ValueType> values{exec, 3};
    gko::array<IndexType> row_ptrs{exec, 3};
    gko::array<IndexType> col_idxs{exec, 3};
    values.fill(gko::one<ValueType>());
    row_ptrs.fill(IndexType{0});
    col_idxs.fill(IndexType{1});

    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto values, auto row_ptrs, auto col_idxs,
                      auto correct) {
            using vt = std::decay_t<decltype(values[0])>;
            using it = std::decay_t<decltype(row_ptrs[0])>;
            gko::matrix::view::csr<vt, it> view{gko::dim<2>{3, 3}, 3, values,
                                                row_ptrs, col_idxs};
            if (view.size == gko::dim<2>(3, 3) &&
                view.num_stored_elements == 3 && view.values == values &&
                view.row_ptrs == row_ptrs && view.col_idxs == col_idxs) {
                *correct = true;
            }
        },
        1, values, row_ptrs, col_idxs, correct);

    ASSERT_TRUE(get_element(correct, 0));
}

TYPED_TEST(CsrView, WorksOnDevice)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    assert_csr_view<value_type, index_type>(this->exec);
}


template <typename ValueIndexType>
class EllView : public CommonTestFixture {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(EllView, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


template <typename ValueType, typename IndexType>
void assert_ell_view(std::shared_ptr<const gko::EXEC_TYPE> exec)
{
    gko::array<bool> correct{exec, {false}};
    gko::array<ValueType> values{exec, 6};
    gko::array<IndexType> col_idxs{exec, 6};
    values.fill(gko::one<ValueType>());
    col_idxs.fill(gko::zero<IndexType>());
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto values, auto col_idxs, auto correct) {
            using device_type = std::decay_t<decltype(values[0])>;
            gko::matrix::view::ell<device_type, IndexType> view{
                gko::dim<2>{2, 3}, 2, 3, values, col_idxs};
            if (view.size == gko::dim<2>(2, 3) &&
                view.num_stored_elements_per_row == 2 && view.stride == 3 &&
                view.values == values && view.col_idxs == col_idxs &&
                &view.val_at(0, 0) == &values[0] &&
                &view.val_at(1, 0) == &values[1] &&
                &view.val_at(1, 1) == &values[4] &&
                view.val_at(1, 1) == gko::one<device_type>() &&
                &view.col_at(0, 0) == &col_idxs[0] &&
                &view.col_at(1, 0) == &col_idxs[1] &&
                &view.col_at(1, 1) == &col_idxs[4] &&
                view.col_at(1, 1) == gko::zero<IndexType>()) {
                *correct = true;
            }
        },
        1, values, col_idxs, correct);
    ASSERT_TRUE(get_element(correct, 0));
}


TYPED_TEST(EllView, WorksOnDevice)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    assert_ell_view<value_type, index_type>(this->exec);
}


template <typename ValueIndexType>
class SellpView : public CommonTestFixture {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(SellpView, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


template <typename ValueType, typename IndexType>
void assert_sellp_view(std::shared_ptr<const gko::EXEC_TYPE> exec)
{
    gko::array<bool> correct{exec, {false}};
    gko::array<ValueType> values{exec, 21};
    gko::array<IndexType> col_idxs{exec, 21};
    gko::array<gko::size_type> slice_lengths{exec, {3, 4}};
    gko::array<gko::size_type> slice_sets{exec, {0, 3, 7}};
    values.fill(gko::one<ValueType>());
    col_idxs.fill(gko::zero<IndexType>());
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto values, auto col_idxs, auto slice_lengths,
                      auto slice_sets, auto correct) {
            using device_type = std::decay_t<decltype(values[0])>;
            gko::matrix::view::sellp<device_type, IndexType> view{
                gko::dim<2>{3, 5}, 2,         3, 7, values, col_idxs,
                slice_lengths,     slice_sets};
            if (view.size == gko::dim<2>(3, 5) && view.slice_size == 2 &&
                view.stride_factor == 3 && view.total_cols == 7 &&
                view.values == values && view.col_idxs == col_idxs &&
                view.slice_lengths == slice_lengths &&
                view.slice_sets == slice_sets &&
                &view.val_at(0, slice_sets[0], 0) == &values[0] &&
                &view.val_at(1, slice_sets[0], 0) == &values[1] &&
                &view.val_at(1, slice_sets[0], 1) == &values[3] &&
                &view.val_at(0, slice_sets[1], 0) == &values[6] &&
                &view.val_at(1, slice_sets[1], 0) == &values[7] &&
                &view.val_at(1, slice_sets[1], 1) == &values[9] &&
                view.val_at(1, slice_sets[1], 1) == gko::one<device_type>() &&
                &view.col_at(0, slice_sets[0], 0) == &col_idxs[0] &&
                &view.col_at(1, slice_sets[0], 0) == &col_idxs[1] &&
                &view.col_at(1, slice_sets[0], 1) == &col_idxs[3] &&
                &view.col_at(0, slice_sets[1], 0) == &col_idxs[6] &&
                &view.col_at(1, slice_sets[1], 0) == &col_idxs[7] &&
                &view.col_at(1, slice_sets[1], 1) == &col_idxs[9] &&
                view.col_at(1, slice_sets[1], 1) == gko::zero<IndexType>()) {
                *correct = true;
            }
        },
        1, values, col_idxs, slice_lengths, slice_sets, correct);
    ASSERT_TRUE(get_element(correct, 0));
}


TYPED_TEST(SellpView, WorksOnDevice)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    assert_sellp_view<value_type, index_type>(this->exec);
}
