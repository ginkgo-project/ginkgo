// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "../tensor.hpp"

#include <memory>

#include <gtest/gtest.h>

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
