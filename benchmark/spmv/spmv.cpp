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

#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <iostream>


#include "benchmark/spmv/spmv_common.hpp"
#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/spmv_validation.hpp"


struct Generator : DefaultSystemGenerator<> {
    void validate_options(const rapidjson::Value& options) const
    {
        if (!options.IsObject() ||
            !((options.HasMember("size") && options.HasMember("stencil")) ||
              options.HasMember("filename"))) {
            std::cerr
                << "Input has to be a JSON array of matrix configurations:\n"
                << example_config << std::endl;
            std::exit(1);
        }
    }
};


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = example_config;
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    run_spmv_benchmark(exec, test_cases, formats, Generator{}, true);

    std::cout << test_cases << std::endl;
}
