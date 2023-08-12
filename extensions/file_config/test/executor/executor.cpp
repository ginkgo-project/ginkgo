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

#include <ginkgo/core/base/executor.hpp>


#include <gtest/gtest.h>
#include <nlohmann/json.hpp>


#include <ginkgo/core/base/exception.hpp>


#include "file_config/file_config.hpp"


TEST(ReferenceExecutor, CreateCorrectExecutor)
{
    auto data = nlohmann::json::parse(R"(
        {"base": "ReferenceExecutor"}
    )");

    auto ptr =
        gko::extensions::file_config::create_from_config<gko::Executor>(data);
    auto exec_ptr = std::dynamic_pointer_cast<gko::ReferenceExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
}


TEST(ReferenceExecutor, ManagerCreateCorrectExecutor)
{
    auto data = nlohmann::json::parse(R"(
        {"base": "ReferenceExecutor"}
    )");
    gko::extensions::file_config::ResourceManager manager;

    auto ptr = manager.build_item<gko::Executor>(data);
    auto exec_ptr = std::dynamic_pointer_cast<gko::ReferenceExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
}


TEST(OmpExecutor, CreateCorrectExecutor)
{
    auto data = nlohmann::json::parse(R"(
        {"base": "OmpExecutor"}
    )");

    auto ptr =
        gko::extensions::file_config::create_from_config<gko::Executor>(data);
    auto exec_ptr = std::dynamic_pointer_cast<gko::OmpExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
}


TEST(OmpExecutor, ManagerCreateCorrectExecutor)
{
    auto data = nlohmann::json::parse(R"(
        {"base": "OmpExecutor"}
    )");

    gko::extensions::file_config::ResourceManager manager;

    auto ptr = manager.build_item<gko::Executor>(data);
    auto exec_ptr = std::dynamic_pointer_cast<gko::OmpExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
}


TEST(CudaExecutor, CreateCorrectExecutor)
{
    auto data = nlohmann::json::parse(R"(
        {"base": "CudaExecutor"}
    )");

    auto ptr =
        gko::extensions::file_config::create_from_config<gko::Executor>(data);
    auto exec_ptr = std::dynamic_pointer_cast<gko::CudaExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 0);
}


TEST(CudaExecutor, CreateCorrectExecutorWithDeviceId)
{
    auto data = nlohmann::json::parse(R"(
        {"base": "CudaExecutor", "device_id": 1}
    )");

    auto ptr =
        gko::extensions::file_config::create_from_config<gko::Executor>(data);
    auto exec_ptr = std::dynamic_pointer_cast<gko::CudaExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 1);
}


TEST(HipExecutor, CreateCorrectExecutor)
{
    auto data = nlohmann::json::parse(R"(
        {"base": "HipExecutor"}
    )");

    auto ptr =
        gko::extensions::file_config::create_from_config<gko::Executor>(data);
    auto exec_ptr = std::dynamic_pointer_cast<gko::HipExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 0);
}


TEST(HipExecutor, CreateCorrectExecutorWithDeviceId)
{
    auto data = nlohmann::json::parse(R"(
        {"base": "HipExecutor", "device_id": 1}
    )");

    auto ptr =
        gko::extensions::file_config::create_from_config<gko::Executor>(data);
    auto exec_ptr = std::dynamic_pointer_cast<gko::HipExecutor>(ptr);

    ASSERT_NE(exec_ptr.get(), nullptr);
    ASSERT_EQ(exec_ptr->get_device_id(), 1);
}


TEST(DpcppExecutor, CreateCorrectExecutor)
{
    if (gko::DpcppExecutor::get_num_devices("all") < 1) {
        GTEST_SKIP() << "Do not contain availble device for dpcpp.";
    } else {
        auto data = nlohmann::json::parse(R"(
            {"base": "DpcppExecutor"}
        )");

        auto ptr =
            gko::extensions::file_config::create_from_config<gko::Executor>(
                data);
        auto exec_ptr = std::dynamic_pointer_cast<gko::DpcppExecutor>(ptr);

        ASSERT_NE(exec_ptr.get(), nullptr);
        ASSERT_EQ(exec_ptr->get_device_id(), 0);
        ASSERT_EQ(exec_ptr->get_device_type(), std::string("all"));
    }
}


TEST(DpcppExecutor, CreateCorrectExecutorWithType)
{
    if (gko::DpcppExecutor::get_num_devices("cpu") < 1) {
        GTEST_SKIP() << "Do not contain availble cpu for dpcpp.";
    } else {
        auto data = nlohmann::json::parse(R"(
            {"base": "DpcppExecutor", "device_type": "cpu"}
        )");

        auto ptr =
            gko::extensions::file_config::create_from_config<gko::Executor>(
                data);
        auto exec_ptr = std::dynamic_pointer_cast<gko::DpcppExecutor>(ptr);

        ASSERT_NE(exec_ptr.get(), nullptr);
        ASSERT_EQ(exec_ptr->get_device_id(), 0);
        ASSERT_EQ(exec_ptr->get_device_type(), std::string("cpu"));
    }
}


TEST(DpcppExecutor, CreateCorrectExecutorWithDeviceIdAndType)
{
    if (gko::DpcppExecutor::get_num_devices("gpu") <= 1) {
        GTEST_SKIP() << "Do not contain enough gpu for this dpcpp test.";
    } else {
        auto data = nlohmann::json::parse(R"(
            {"base": "DpcppExecutor", "device_type": "gpu", "device_id": 1}
        )");

        auto ptr =
            gko::extensions::file_config::create_from_config<gko::Executor>(
                data);
        auto exec_ptr = std::dynamic_pointer_cast<gko::DpcppExecutor>(ptr);

        ASSERT_NE(exec_ptr.get(), nullptr);
        ASSERT_EQ(exec_ptr->get_device_id(), 1);
        ASSERT_EQ(exec_ptr->get_device_type(), std::string("gpu"));
    }
}
