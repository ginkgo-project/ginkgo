/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include <rapidjson/document.h>


#include <ginkgo/core/preconditioner/isai.hpp>


#include "resource_manager/resource_manager.hpp"


namespace {


TEST(IsaiFactory, CreateStandAlone)
{
    const char json[] =
        "{\"base\": \"IsaiFactory\",\
          \"IsaiType\": \"isai_lower\",\
          \"exec\": {\"base\": \"ReferenceExecutor\"}\
         }";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::LinOpFactory>(
            d);
    auto preconfactory_ptr =
        std::dynamic_pointer_cast<gko::preconditioner::Isai<
            gko::preconditioner::isai_type::lower, double, int>::Factory>(ptr);

    ASSERT_NE(preconfactory_ptr.get(), nullptr);
    ASSERT_EQ(preconfactory_ptr->get_parameters().skip_sorting, false);
    ASSERT_EQ(preconfactory_ptr->get_parameters().sparsity_power, 1);
    ASSERT_EQ(preconfactory_ptr->get_parameters().excess_limit, 0u);
}


}  // namespace
