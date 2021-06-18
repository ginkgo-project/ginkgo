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

#include <ginkgo/core/base/executor.hpp>


#include <thread>
#include <type_traits>


#if defined(__unix__) || defined(__APPLE__)
#include <utmpx.h>
#endif


#include <gtest/gtest.h>
#include "rapidjson/document.h"

#include <ginkgo/core/base/exception.hpp>
#include <resource_manager/resource_manager.hpp>


namespace {


TEST(Dense, DenseStandAlone)
{
    using type = gko::matrix::Dense<double>;
    const char json[] =
        "{\"base\": \"Dense\", \"exec\": {\"base\": \"ReferenceExecutor\"}}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::LinOp>(d);
    auto mtx_ptr = std::dynamic_pointer_cast<type>(ptr);

    ASSERT_NE(mtx_ptr.get(), nullptr);
    ASSERT_EQ(mtx_ptr->get_size(), gko::dim<2>{});
    ASSERT_EQ(mtx_ptr->get_stride(), 0);
}


TEST(Dense, DenseStandAloneWithSize)
{
    using type = gko::matrix::Dense<double>;
    const char json[] =
        "{\"base\": \"Dense\", \"dim\": [3, 4], \"exec\": {\"base\": "
        "\"ReferenceExecutor\"}}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::LinOp>(d);
    auto mtx_ptr = std::dynamic_pointer_cast<type>(ptr);

    ASSERT_NE(mtx_ptr.get(), nullptr);
    ASSERT_EQ(mtx_ptr->get_size(), gko::dim<2>(3, 4));
    ASSERT_EQ(mtx_ptr->get_stride(), 4);
}


TEST(Dense, DenseStandAloneWithSizeStride)
{
    using type = gko::matrix::Dense<double>;
    const char json[] =
        "{\"base\": \"Dense\", \"dim\": [3, 4], \"stride\": 10, \"exec\": "
        "{\"base\": \"ReferenceExecutor\"}}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::LinOp>(d);
    auto mtx_ptr = std::dynamic_pointer_cast<type>(ptr);

    ASSERT_NE(mtx_ptr.get(), nullptr);
    ASSERT_EQ(mtx_ptr->get_size(), gko::dim<2>(3, 4));
    ASSERT_EQ(mtx_ptr->get_stride(), 10);
}

TEST(Dense, DenseStandAloneWithSizeStrideInheritExecutor)
{
    using type = gko::matrix::Dense<double>;
    const char json[] =
        "{\"base\": \"Dense\", \"dim\": [3, 4], \"stride\": 10, \"exec\": "
        "\"inherit\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);
    auto exec = share(gko::ReferenceExecutor::create());

    auto ptr = gko::extension::resource_manager::create_from_config<gko::LinOp>(
        d, exec);
    auto mtx_ptr = std::dynamic_pointer_cast<type>(ptr);

    ASSERT_NE(mtx_ptr.get(), nullptr);
    ASSERT_EQ(mtx_ptr->get_size(), gko::dim<2>(3, 4));
    ASSERT_EQ(mtx_ptr->get_stride(), 10);
    ASSERT_EQ(mtx_ptr->get_executor().get(), exec.get());
}

TEST(Dense, DenseManagerWithSizeStrideExecutor)
{
    using type = gko::matrix::Dense<double>;
    const char json[] =
        "[{\"name\": \"ref\", \"base\": \"ReferenceExecutor\"}, \
          {\"name\": \"mtx\", \"base\": \"Dense\", \
           \"dim\": [3, 4], \"stride\": 10, \
           \"exec\": \"ref\"}]";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);
    auto exec = share(gko::ReferenceExecutor::create());
    gko::extension::resource_manager::ResourceManager manager;

    manager.read(d);
    auto mtx_ptr = manager.search_data<type>("mtx");
    auto exec_ptr = manager.search_data<gko::Executor>("ref");

    ASSERT_NE(mtx_ptr.get(), nullptr);
    ASSERT_EQ(mtx_ptr->get_size(), gko::dim<2>(3, 4));
    ASSERT_EQ(mtx_ptr->get_stride(), 10);
    ASSERT_EQ(mtx_ptr->get_executor().get(), exec_ptr.get());
}


// TEST(Dense, DenseManagerWithSizeStrideExecutorbutInherit)
// {
//     using type = gko::matrix::Dense<double>;
//     const char json[] =
//         "[{\"name\": \"ref\", \"base\": \"ReferenceExecutor\"},\
//           {\"base\": \"Dense\", \"dim\": [3, 4], \"stride\": 10, \"exec\":
//           \"inherit\"}]";
//     rapidjson::StringStream s(json);
//     rapidjson::Document d;
//     d.ParseStream(s);
//     auto exec = share(gko::ReferenceExecutor::create());
//     gko::extension::resource_manager::ResourceManager manager;

//     auto ptr = manager.build_item<gko::LinOp>(d, exec);
//     auto mtx_ptr = std::dynamic_pointer_cast<type>(ptr);

//     ASSERT_NE(mtx_ptr.get(), nullptr);
//     ASSERT_EQ(mtx_ptr->get_size(), gko::dim<2>(3, 4));
//     ASSERT_EQ(mtx_ptr->get_stride(), 10);
//     ASSERT_EQ(mtx_ptr->get_executor().get(), exec.get());
// }


}  // namespace
