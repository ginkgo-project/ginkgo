/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/lin_op.hpp>


#include <gtest/gtest.h>


namespace {


class FactoryParameter : public ::testing::Test {
protected:
    FactoryParameter() {}

public:
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
