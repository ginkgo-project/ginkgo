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

#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>


#include "benchmark/utils/general.hpp"


// some Ginkgo shortcuts
using etype = double;


namespace {
std::string input_format =
    "  [\n"
    "    {\n"
    "      \"filename\": \"<output-file>\",\n"
    "      \"problem\": {\n"
    "        \"type\": <matrix-type>\",\n"
    "        \"num_blocks\": <num-blocks>,\n"
    "        \"block_size\": <block-size>\n"
    "      }\n"
    "    },\n"
    "    ...\n"
    "  ]\n"
    "  <output-file> is a string specifying a path to the output file\n"
    "  <matrix-type> is a string specifying the type of matrix to generate,\n"
    "    supported values are \"block-diagonal\"\n"
    "  All other properties are optional, depending on <matrix-type>\n"
    "  Properties for \"block-diagonal\":\n"
    "    <num-blocks> is the number of dense diagonal blocks\n"
    "    <block-size> is the size of each dense block\n"
    "    The generated matrix will have a dense block of size <block-size>,\n"
    "    with random real values chosen uniformly in the interval [-1, 1],\n"
    "    repeated <num-blocks> times on the diagonal.\n";
}  // namespace


// clang-format off
// input validation
[[noreturn]] void print_config_error_and_exit(int code = 1)
{
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << input_format << std::endl;
    std::exit(code);
}
// clang-format on


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("filename") ||
        !value["filename"].IsString() || !value.HasMember("problem") ||
        !value["problem"].IsObject() || !value["problem"].HasMember("type") ||
        !value["problem"]["type"].IsString()) {
        print_config_error_and_exit(2);
    }
}


using generator_function =
    std::function<gko::matrix_data<etype>(rapidjson::Value &, std::ranlux24 &)>;


// matrix generators
gko::matrix_data<etype> generate_block_diagonal(rapidjson::Value &config,
                                                std::ranlux24 &engine)
{
    if (!config.HasMember("num_blocks") || !config["num_blocks"].IsUint() ||
        !config.HasMember("block_size") || !config["block_size"].IsUint()) {
        print_config_error_and_exit(2);
    }
    auto num_blocks = config["num_blocks"].GetUint();
    auto block_size = config["block_size"].GetUint();
    auto block = gko::matrix_data<etype>(
        gko::dim<2>(block_size), std::uniform_real_distribution<>(-1.0, 1.0),
        engine);
    return gko::matrix_data<etype>::diag(num_blocks, block);
}


// generator mapping
std::map<std::string, generator_function> generator{
    {"block-diagonal", generate_block_diagonal}};


int main(int argc, char *argv[])
{
    std::string header =
        "A utility that generates various types of "
        "matrices.\n";
    initialize_argument_parsing(&argc, &argv, header, input_format);

    std::clog << gko::version_info::get() << std::endl;

    auto engine = get_engine();
    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document configurations;
    configurations.ParseStream(jcin);

    if (!configurations.IsArray()) {
        print_config_error_and_exit(1);
    }

    for (auto &config : configurations.GetArray()) {
        try {
            validate_option_object(config);
            std::clog << "Generating matrix: " << config << std::endl;
            auto filename = config["filename"].GetString();
            auto type = config["problem"]["type"].GetString();
            auto mdata = generator[type](config["problem"], engine);
            std::ofstream ofs(filename);
            gko::write_raw(ofs, mdata, gko::layout_type::coordinate);
        } catch (const std::exception &e) {
            std::cerr << "Error generating matrix, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << configurations << std::endl;
}
