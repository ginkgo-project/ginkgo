// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


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


void validate_option_object(const json& value)
{
    if (!value.is_object() || !value.contains("filename") ||
        !value["filename"].is_string() || !value.contains("problem") ||
        !value["problem"].is_object() || !value["problem"].contains("type") ||
        !value["problem"]["type"].is_string()) {
        print_config_error_and_exit(2);
    }
}


using generator_function = std::function<gko::matrix_data<etype, itype>(
    json&, std::default_random_engine&)>;


// matrix generators
gko::matrix_data<etype, itype> generate_block_diagonal(
    json& config, std::default_random_engine& engine)
{
    if (!config.contains("num_blocks") ||
        !config["num_blocks"].is_number_unsigned() ||
        !config.contains("block_size") ||
        !config["block_size"].is_number_unsigned()) {
        print_config_error_and_exit(2);
    }
    auto num_blocks = config["num_blocks"].get<gko::uint64>();
    auto block_size = config["block_size"].get<gko::uint64>();
    auto block = gko::matrix_data<etype, itype>(
        gko::dim<2>(block_size),
        std::uniform_real_distribution<rc_etype>(-1.0, 1.0), engine);
    return gko::matrix_data<etype, itype>::diag(gko::dim<2>{num_blocks}, block);
}


// generator mapping
std::map<std::string, generator_function> generator{
    {"block-diagonal", generate_block_diagonal}};


int main(int argc, char* argv[])
{
    std::string header =
        "A utility that generates various types of "
        "matrices.\n";
    initialize_argument_parsing(&argc, &argv, header, input_format);

    std::clog << gko::version_info::get() << std::endl;

    auto engine = get_engine();
    auto configurations = json::parse(get_input_stream());

    if (!configurations.is_array()) {
        print_config_error_and_exit(1);
    }

    for (auto& config : configurations) {
        try {
            validate_option_object(config);
            std::clog << "Generating matrix: " << config << std::endl;
            auto filename = config["filename"].get<std::string>();
            auto type = config["problem"]["type"].get<std::string>();
            auto mdata = generator[type](config["problem"], engine);
            std::ofstream ofs(filename);
            gko::write_raw(ofs, mdata, gko::layout_type::coordinate);
        } catch (const std::exception& e) {
            std::cerr << "Error generating matrix, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << configurations << std::endl;
}
