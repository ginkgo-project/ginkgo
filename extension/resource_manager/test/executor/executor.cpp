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
// #include <resource_manager/base/generic_constructor.hpp>
#include <resource_manager/executor/executor.hpp>
#include <resource_manager/resource_manager.hpp>


namespace {


TEST(ReferenceExecutor, CreateCorrectExecutor)
{
    const char json[] = "{\"base\": \"ReferenceExecutor\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::ReferenceExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
}


TEST(ReferenceExecutor, ManagerCreateCorrectExecutor)
{
    const char json[] = "{\"base\": \"ReferenceExecutor\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);
    gko::extension::resource_manager::ResourceManager manager;

    auto ptr = manager.build_item<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::ReferenceExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
}


TEST(OmpExecutor, CreateCorrectExecutor)
{
    const char json[] = "{\"base\": \"OmpExecutor\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::OmpExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
}


TEST(OmpExecutor, ManagerCreateCorrectExecutor)
{
    const char json[] = "{\"base\": \"OmpExecutor\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);
    gko::extension::resource_manager::ResourceManager manager;

    auto ptr = manager.build_item<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::OmpExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
}


TEST(CudaExecutor, CreateCorrectExecutor)
{
    const char json[] = "{\"base\": \"CudaExecutor\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::CudaExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 0);
}


TEST(CudaExecutor, CreateCorrectExecutorWithDeviceId)
{
    const char json[] = "{\"base\": \"CudaExecutor\", \"device_id\": 1}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::CudaExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 1);
}


TEST(HipExecutor, CreateCorrectExecutor)
{
    const char json[] = "{\"base\": \"HipExecutor\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::HipExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 0);
}


TEST(HipExecutor, CreateCorrectExecutorWithDeviceId)
{
    const char json[] = "{\"base\": \"HipExecutor\", \"device_id\": 1}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::HipExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 1);
}


TEST(DpcppExecutor, CreateCorrectExecutor)
{
    const char json[] = "{\"base\": \"DpcppExecutor\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::DpcppExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 0);
    ASSERT_EQ(exec_ptr->get_device_type(), std::string("all"));
}


TEST(DpcppExecutor, CreateCorrectExecutorWithDeviceIdAndType)
{
    const char json[] =
        "{\"base\": \"DpcppExecutor\", \"device_id\": 1, \"device_type\": "
        "\"gpu\"}";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::Executor>(d);
    auto exec_ptr = std::dynamic_pointer_cast<gko::DpcppExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 1);
    ASSERT_EQ(exec_ptr->get_device_type(), std::string("gpu"));
}


}  // namespace
