// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>


#include <gtest/gtest.h>


namespace {


class FactoryParameter : public ::testing::Test {
protected:
    FactoryParameter() {}

public:
    // FACTORY_PARAMETER macro needs self, which is usually available in
    // enable_parameters_type. To reduce complexity, we add self here.
    GKO_ENABLE_SELF(FactoryParameter);

    std::vector<int> GKO_FACTORY_PARAMETER_VECTOR(vector_parameter, 10, 11);
    int GKO_FACTORY_PARAMETER_SCALAR(scalar_parameter, -4);
};


TEST_F(FactoryParameter, WorksOnHipDefault)
{
    std::vector<int> expected{10, 11};

    ASSERT_EQ(vector_parameter, expected);
    ASSERT_EQ(scalar_parameter, -4);
}


TEST_F(FactoryParameter, WorksOnHip0)
{
    std::vector<int> expected{};

    auto result = &this->with_vector_parameter();

    ASSERT_EQ(vector_parameter, expected);
    ASSERT_EQ(result, this);
}


TEST_F(FactoryParameter, WorksOnHip1)
{
    std::vector<int> expected{2};

    this->with_vector_parameter(2).with_scalar_parameter(3);

    ASSERT_EQ(vector_parameter, expected);
    ASSERT_EQ(scalar_parameter, 3);
}


TEST_F(FactoryParameter, WorksOnHip2)
{
    std::vector<int> expected{8, 3};

    this->with_vector_parameter(8, 3);

    ASSERT_EQ(vector_parameter, expected);
}


TEST_F(FactoryParameter, WorksOnHip3)
{
    std::vector<int> expected{1, 7, 2};

    this->with_vector_parameter(1, 7, 2);

    ASSERT_EQ(vector_parameter, expected);
}


TEST_F(FactoryParameter, WorksOnHip4)
{
    std::vector<int> expected{4, 5, 4, 2};

    this->with_vector_parameter(4, 5, 4, 2);

    ASSERT_EQ(vector_parameter, expected);
}


TEST_F(FactoryParameter, WorksOnHip5)
{
    std::vector<int> expected{9, 3, 4, 2, 7};

    this->with_vector_parameter(9, 3, 4, 2, 7);

    ASSERT_EQ(vector_parameter, expected);
}


}  // namespace
