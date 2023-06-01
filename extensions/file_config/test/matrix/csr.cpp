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

#include <memory>


#include <gtest/gtest.h>
#include <nlohmann/json.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "file_config/base/type_string.hpp"
#include "file_config/file_config.hpp"


namespace {


using ValueIndexTypes = ::testing::Types<
    std::tuple<float, gko::int32>, std::tuple<double, gko::int32>,
    std::tuple<std::complex<float>, gko::int32>,
    std::tuple<std::complex<double>, gko::int32>, std::tuple<float, gko::int64>,
    std::tuple<double, gko::int64>, std::tuple<std::complex<float>, gko::int64>,
    std::tuple<std::complex<double>, gko::int64>>;

template <typename ValueIndexType>
class Csr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

    Csr() : exec(gko::ReferenceExecutor::create())
    {
        exec_json = {{"base", "ReferenceExecutor"}};
        base_str =
            "Csr<" + gko::extensions::file_config::get_string<value_type>() +
            ", " + gko::extensions::file_config::get_string<index_type>() + ">";
    }

public:
    void equal_to_mtx(std::shared_ptr<const gko::LinOp> result,
                      const Mtx* answer)
    {
        auto mtx_ptr = std::dynamic_pointer_cast<const Mtx>(result);
        ASSERT_NE(mtx_ptr, nullptr);
        ASSERT_EQ(mtx_ptr->get_size(), answer->get_size());
        ASSERT_EQ(mtx_ptr->get_num_stored_elements(),
                  answer->get_num_stored_elements());
        ASSERT_EQ(mtx_ptr->get_strategy()->get_name(),
                  answer->get_strategy()->get_name());
    }

    template <typename BuildType>
    void run(const nlohmann::json& config, const Mtx* answer)
    {
        {
            SCOPED_TRACE("Standalone");
            nlohmann::json data = config;
            data["exec"] = exec_json;
            data["base"] = base_str;

            auto ptr =
                gko::extensions::file_config::create_from_config<BuildType>(
                    data);

            this->equal_to_mtx(ptr, answer);
            ASSERT_TRUE((std::is_same<typename decltype(ptr)::element_type,
                                      BuildType>::value));
        }
        {
            SCOPED_TRACE("Standalone With Input");
            nlohmann::json data = config;
            data["exec"] = "inherit";
            data["base"] = base_str;

            auto ptr =
                gko::extensions::file_config::create_from_config<BuildType>(
                    data, exec);

            this->equal_to_mtx(ptr, answer);
            ASSERT_TRUE((std::is_same<typename decltype(ptr)::element_type,
                                      BuildType>::value));
            ASSERT_EQ(ptr->get_executor(), answer->get_executor());
        }
        {
            SCOPED_TRACE("Default Template");
            if (std::is_same<gko::matrix::Csr<>, Mtx>::value) {
                {
                    SCOPED_TRACE("Without <>");
                    nlohmann::json data = config;
                    data["base"] = "Csr";
                    data["exec"] = exec_json;

                    auto ptr = gko::extensions::file_config::create_from_config<
                        BuildType>(data);

                    this->equal_to_mtx(ptr, answer);
                    ASSERT_TRUE(
                        (std::is_same<typename decltype(ptr)::element_type,
                                      BuildType>::value));
                }
                {
                    SCOPED_TRACE("With <>");
                    nlohmann::json data = config;
                    data["base"] = "Csr<>";
                    data["exec"] = exec_json;

                    auto ptr = gko::extensions::file_config::create_from_config<
                        BuildType>(data);

                    this->equal_to_mtx(ptr, answer);
                    ASSERT_TRUE(
                        (std::is_same<typename decltype(ptr)::element_type,
                                      BuildType>::value));
                }
            }
        }
        {
            SCOPED_TRACE("Default IndexType");
            if (std::is_same<gko::matrix::Csr<value_type>, Mtx>::value) {
                nlohmann::json data = config;
                data["base"] =
                    "Csr<" +
                    gko::extensions::file_config::get_string<value_type>() +
                    ">";
                data["exec"] = exec_json;

                auto ptr =
                    gko::extensions::file_config::create_from_config<BuildType>(
                        data);

                this->equal_to_mtx(ptr, answer);
                ASSERT_TRUE((std::is_same<typename decltype(ptr)::element_type,
                                          BuildType>::value));
            }
        }
        {
            SCOPED_TRACE("ResourceManager Standalone");
            nlohmann::json data = config;
            data["exec"] = exec_json;
            data["base"] = base_str;
            gko::extensions::file_config::ResourceManager manager;

            auto ptr = manager.build_item<BuildType>(data);

            this->equal_to_mtx(ptr, answer);
            ASSERT_TRUE((std::is_same<typename decltype(ptr)::element_type,
                                      BuildType>::value));
        }
        {
            SCOPED_TRACE("ResourceManager");
            nlohmann::json tmp = config;
            tmp["exec"] = "ref";
            tmp["name"] = "matrix";
            tmp["base"] = base_str;
            auto exec_item = nlohmann::json::parse(R"({
                "name": "ref",
                "base": "ReferenceExecutor"
            })");
            nlohmann::json data;
            data.push_back(exec_item);
            data.push_back(tmp);
            gko::extensions::file_config::ResourceManager manager;

            manager.read(data);
            auto ptr = manager.search_data<BuildType>("matrix");

            this->equal_to_mtx(ptr, answer);
            ASSERT_TRUE((std::is_same<typename decltype(ptr)::element_type,
                                      BuildType>::value));
        }
        {
            SCOPED_TRACE("ResourceManager With Input");
            nlohmann::json data = config;
            data["base"] = base_str;
            data["exec"] = "inherit";
            gko::extensions::file_config::ResourceManager manager;

            auto ptr = manager.build_item<BuildType>(data, exec);

            this->equal_to_mtx(ptr, answer);
            ASSERT_EQ(ptr->get_executor(), answer->get_executor());
            ASSERT_TRUE((std::is_same<typename decltype(ptr)::element_type,
                                      BuildType>::value));
        }
        {
            SCOPED_TRACE("ResourceManager With Storage");
            nlohmann::json data = config;
            data["base"] = base_str;
            data["exec"] = "ref";
            gko::extensions::file_config::ResourceManager manager;

            manager.insert_data("ref", exec);
            auto ptr = manager.build_item<BuildType>(data);

            this->equal_to_mtx(ptr, answer);
            ASSERT_EQ(ptr->get_executor(), answer->get_executor());
            ASSERT_TRUE((std::is_same<typename decltype(ptr)::element_type,
                                      BuildType>::value));
        }
    }
    std::shared_ptr<gko::ReferenceExecutor> exec;
    nlohmann::json exec_json;
    std::string base_str;
};

TYPED_TEST_SUITE(Csr, ValueIndexTypes);

TYPED_TEST(Csr, Exec)
{
    nlohmann::json data;
    auto answer = TestFixture::Mtx::create(this->exec);

    this->template run<gko::LinOp>(data, answer.get());
    this->template run<typename TestFixture::Mtx>(data, answer.get());
}


TYPED_TEST(Csr, ExecDim)
{
    nlohmann::json data = nlohmann::json::parse(R"({
        "dim": [3, 4]
    })");
    auto answer = TestFixture::Mtx::create(this->exec, gko::dim<2>(3, 4));

    this->template run<gko::LinOp>(data, answer.get());
    this->template run<typename TestFixture::Mtx>(data, answer.get());
}


TYPED_TEST(Csr, ExecSquareDim)
{
    nlohmann::json data = nlohmann::json::parse(R"({
        "dim": 3
    })");
    auto answer = TestFixture::Mtx::create(this->exec, gko::dim<2>(3, 3));

    this->template run<gko::LinOp>(data, answer.get());
    this->template run<typename TestFixture::Mtx>(data, answer.get());
}


TYPED_TEST(Csr, ExecDimElements)
{
    nlohmann::json data = nlohmann::json::parse(R"({
        "dim": [3, 4],
        "num_nonzeros": 10
    })");
    auto answer = TestFixture::Mtx::create(this->exec, gko::dim<2>(3, 4), 10);

    this->template run<gko::LinOp>(data, answer.get());
    this->template run<typename TestFixture::Mtx>(data, answer.get());
}


TYPED_TEST(Csr, ExecDimElementsStrategySparselib)
{
    using Mtx = typename TestFixture::Mtx;
    nlohmann::json data = nlohmann::json::parse(R"({
        "dim": [3, 4],
        "num_nonzeros": 10,
        "strategy": "sparselib"
    })");
    auto answer = Mtx::create(this->exec, gko::dim<2>(3, 4), 10,
                              std::make_shared<typename Mtx::sparselib>());

    this->template run<gko::LinOp>(data, answer.get());
    this->template run<Mtx>(data, answer.get());
}


TYPED_TEST(Csr, ExecStrategySparselib)
{
    using Mtx = typename TestFixture::Mtx;
    nlohmann::json data = nlohmann::json::parse(R"({
        "strategy": "sparselib"
    })");
    auto answer =
        Mtx::create(this->exec, std::make_shared<typename Mtx::sparselib>());

    this->template run<gko::LinOp>(data, answer.get());
    this->template run<Mtx>(data, answer.get());
}


class CsrStrategy : public ::testing::Test {
public:
    using Mtx = gko::matrix::Csr<float>;

    template <typename StrategyType>
    void run(std::string executor, std::string input_strategy)
    {
        SCOPED_TRACE(input_strategy);
        nlohmann::json data;
        data["base"] = "Csr<float>";
        data["exec"] = {{"base", executor}};
        data["strategy"] = input_strategy;
        auto ptr = gko::extensions::file_config::create_from_config<
            gko::matrix::Csr<float>>(data);
        ASSERT_NE(std::dynamic_pointer_cast<StrategyType>(ptr->get_strategy()),
                  nullptr);
    }

    bool module_is_compiled(std::string module_name)
    {
        auto version = gko::version_info::get();
        static const std::string not_compiled_tag = "not compiled";
        if (module_name == "omp") {
            return version.omp_version.tag != not_compiled_tag;
        } else if (module_name == "cuda") {
            return version.cuda_version.tag != not_compiled_tag;
        } else if (module_name == "hip") {
            return version.hip_version.tag != not_compiled_tag;
        } else if (module_name == "dpcpp") {
            return version.dpcpp_version.tag != not_compiled_tag;
        }
        return false;
    }
};


TEST_F(CsrStrategy, ReferenceExecutor)
{
    this->template run<typename Mtx::sparselib>("ReferenceExecutor",
                                                "sparselib");
    this->template run<typename Mtx::sparselib>("ReferenceExecutor",
                                                "cusparse");
    this->template run<typename Mtx::merge_path>("ReferenceExecutor",
                                                 "merge_path");
    this->template run<typename Mtx::classical>("ReferenceExecutor",
                                                "classical");
    // automatical and load_balance will use classical on ReferenceExecutor
    this->template run<typename Mtx::classical>("ReferenceExecutor",
                                                "automatical");
    this->template run<typename Mtx::classical>("ReferenceExecutor",
                                                "load_balance");
}


TEST_F(CsrStrategy, OmpExecutor)
{
    if (!this->module_is_compiled("omp")) {
        GTEST_SKIP() << "Omp module is not compiled.";
    }
    this->template run<typename Mtx::sparselib>("OmpExecutor", "sparselib");
    this->template run<typename Mtx::sparselib>("OmpExecutor", "cusparse");
    this->template run<typename Mtx::merge_path>("OmpExecutor", "merge_path");
    this->template run<typename Mtx::classical>("OmpExecutor", "classical");
    // automatical and load_balance will use classical on OmpExecutor
    this->template run<typename Mtx::classical>("OmpExecutor", "automatical");
    this->template run<typename Mtx::classical>("OmpExecutor", "load_balance");
}


TEST_F(CsrStrategy, CudaExecutor)
{
    if (!this->module_is_compiled("cuda")) {
        GTEST_SKIP() << "Cuda module is not compiled.";
    }
    this->template run<typename Mtx::sparselib>("CudaExecutor", "sparselib");
    this->template run<typename Mtx::sparselib>("CudaExecutor", "cusparse");
    this->template run<typename Mtx::merge_path>("CudaExecutor", "merge_path");
    this->template run<typename Mtx::classical>("CudaExecutor", "classical");
    this->template run<typename Mtx::automatical>("CudaExecutor",
                                                  "automatical");
    this->template run<typename Mtx::load_balance>("CudaExecutor",
                                                   "load_balance");
}


TEST_F(CsrStrategy, HipExecutor)
{
    if (!this->module_is_compiled("hip")) {
        GTEST_SKIP() << "Hip module is not compiled.";
    }
    this->template run<typename Mtx::sparselib>("HipExecutor", "sparselib");
    this->template run<typename Mtx::sparselib>("HipExecutor", "cusparse");
    this->template run<typename Mtx::merge_path>("HipExecutor", "merge_path");
    this->template run<typename Mtx::classical>("HipExecutor", "classical");
    this->template run<typename Mtx::automatical>("HipExecutor", "automatical");
    this->template run<typename Mtx::load_balance>("HipExecutor",
                                                   "load_balance");
}


TEST_F(CsrStrategy, DpcppExecutor)
{
    if (!this->module_is_compiled("dpcpp")) {
        GTEST_SKIP() << "Dpcpp module is not compiled.";
    }
    this->template run<typename Mtx::sparselib>("DpcppExecutor", "sparselib");
    this->template run<typename Mtx::sparselib>("DpcppExecutor", "cusparse");
    this->template run<typename Mtx::merge_path>("DpcppExecutor", "merge_path");
    this->template run<typename Mtx::classical>("DpcppExecutor", "classical");
    this->template run<typename Mtx::automatical>("DpcppExecutor",
                                                  "automatical");
    this->template run<typename Mtx::load_balance>("DpcppExecutor",
                                                   "load_balance");
}


}  // namespace
