// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../tensor.hpp"

#include <memory>

#include <gtest/gtest.h>

#include "core/matrix/batch_struct.hpp"
#include "core/test/utils.hpp"

auto exec = gko::ReferenceExecutor::create();

TEST(Tensor, CanCreateEmpty)
{
    auto tensor = std::make_unique<tensor::TensorLeft>(exec);

    ASSERT_EQ(tensor->get_size(), gko::batch_dim<2>{});
    ASSERT_EQ(tensor->get_executor(), exec);
}

TEST(Tensor, CanCreateWithSize)
{
    auto tensor = std::make_unique<tensor::TensorLeft>(exec, 3, 4);

    auto expected_size = gko::batch_dim<2>{3, gko::dim<2>{64, 64}};
    ASSERT_EQ(tensor->get_size(), expected_size);
}

TEST(Tensor, CanCreateFromData)
{
    auto data = gko::batch::matrix::Dense<tensor::value_type>::create(
        exec, gko::batch_dim<2>{3, gko::dim<2>{4, 4}});
    for (auto i = 0; i < data->get_num_batch_items(); ++i) {
        data->create_view_for_item(i)->fill(i + 1);
    }
    auto orig = gko::clone(data);

    auto tensor = std::make_unique<tensor::TensorLeft>(std::move(data));

    auto expected_size = gko::batch_dim<2>{3, gko::dim<2>{64, 64}};
    ASSERT_EQ(tensor->get_size(), expected_size);
    auto view = tensor->create_view();
    auto tensor_data = gko::batch::matrix::Dense<tensor::value_type>::create(
        exec, orig->get_size(),
        gko::make_array_view(exec, orig->get_num_stored_elements(),
                             const_cast<tensor::value_type*>(view.data)));
    GKO_ASSERT_BATCH_MTX_NEAR(tensor_data, orig, 0.0);
}


TEST(TensorConvert, CanConvertDenseId)
{
    auto A = gko::initialize<gko::matrix::Dense<tensor::value_type>>(
        {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, exec);
    auto Id = gko::matrix::Identity<tensor::value_type>::create(exec, 2);
    auto result =
        gko::matrix::Dense<tensor::value_type>::create(exec, gko::dim<2>{6, 4});

    tensor::convert_tensor(A, Id, result);

    auto expected =
        gko::initialize<gko::matrix::Dense<tensor::value_type>>({{1, 0, 2, 0},
                                                                 {0, 1, 0, 2},
                                                                 {3, 0, 4, 0},
                                                                 {0, 3, 0, 4},
                                                                 {5, 0, 6, 0},
                                                                 {0, 5, 0, 6}},
                                                                exec);
    GKO_ASSERT_MTX_NEAR(result, expected, 0.0);
}

TEST(TensorConvert, CanConvertIdDense)
{
    auto A = gko::initialize<gko::matrix::Dense<tensor::value_type>>(
        {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, exec);
    auto Id = gko::matrix::Identity<tensor::value_type>::create(exec, 2);
    auto result =
        gko::matrix::Dense<tensor::value_type>::create(exec, gko::dim<2>{6, 4});

    tensor::convert_tensor(Id, A, result);

    auto expected =
        gko::initialize<gko::matrix::Dense<tensor::value_type>>({{1, 2, 0, 0},
                                                                 {3, 4, 0, 0},
                                                                 {5, 6, 0, 0},
                                                                 {0, 0, 1, 2},
                                                                 {0, 0, 3, 4},
                                                                 {0, 0, 5, 6}},
                                                                exec);
    GKO_ASSERT_MTX_NEAR(result, expected, 0.0);
}


class Tensor2 : public testing::Test {
public:
    Tensor2()
    {
        auto data = gko::batch::matrix::Dense<tensor::value_type>::create(
            exec, gko::batch_dim<2>{3, gko::dim<2>{4, 4}});
        for (auto i = 0; i < data->get_num_batch_items(); ++i) {
            data->create_view_for_item(i)->fill(i + 1);
        }

        tensor = std::make_unique<tensor::TensorLeft>(std::move(data));

        auto num_rows = tensor->get_common_size()[0];
        auto vector_size = gko::batch_dim<2>{tensor->get_num_batch_items(),
                                             gko::dim<2>{num_rows, 1}};
        x = gko::batch::MultiVector<tensor::value_type>::create(exec,
                                                                vector_size);
        b = gko::batch::MultiVector<tensor::value_type>::create(exec,
                                                                vector_size);
        for (gko::size_type batch = 0; batch < x->get_num_batch_items();
             ++batch) {
            x->create_view_for_item(batch)->read(
                gko::test::generate_random_matrix_data<tensor::value_type,
                                                       gko::int32>(
                    vector_size.get_common_size()[0],
                    vector_size.get_common_size()[1],
                    std::uniform_int_distribution<>(1, 1),
                    std::uniform_real_distribution<>(), engine));
            b->create_view_for_item(batch)->fill(0.0);
        }
    }

    std::default_random_engine engine{42};

    std::unique_ptr<tensor::TensorLeft> tensor;
    std::unique_ptr<gko::batch::MultiVector<tensor::value_type>> x;
    std::unique_ptr<gko::batch::MultiVector<tensor::value_type>> b;
};

TEST_F(Tensor2, CanConvert)
{
    auto mat = convert(tensor);

    ASSERT_EQ(mat->get_size(), tensor->get_size());
    gko::write(std::ofstream("batch.mtx"), mat->create_view_for_item(1));
}


TEST_F(Tensor2, CanApply)
{
    gko::size_type batch_id = 1;
    auto view = tensor->create_view();
    auto item = tensor::extract_batch_item(view, batch_id);
    auto x_view = gko::batch::to_const(x->create_view());
    auto b_view = b->create_view();

    tensor::simple_apply(item, gko::batch::extract_batch_item(x_view, batch_id),
                         gko::batch::extract_batch_item(b_view, batch_id),
                         gko::reference_kernel{});

    auto dense = convert(tensor);
    auto expected_b = gko::clone(b);
    dense->apply(x, expected_b);
    GKO_ASSERT_MTX_NEAR(b->create_view_for_item(batch_id).get(),
                        expected_b->create_view_for_item(batch_id).get(),
                        r<tensor::value_type>::value);
}
