// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


using generator_function = std::function<gko::matrix_data<etype, itype>(
    json&, std::default_random_engine&)>;


// matrix generators
gko::matrix_data<etype, itype> generate_block_diagonal(
    json& config, std::default_random_engine& engine)
{
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

    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/matrix-generator.json"));

    initialize_argument_parsing(&argc, &argv, header, schema["examples"]);

    std::clog << gko::version_info::get() << std::endl;

    auto engine = get_engine();
    auto configurations = json::parse(get_input_stream());

    json_schema::json_validator validator(json_loader);  // create validator

    try {
        validator.set_root_schema(schema);  // insert root-schema
    } catch (const std::exception& e) {
        std::cerr << "Validation of schema failed, here is why: " << e.what()
                  << "\n";
        return EXIT_FAILURE;
    }
    try {
        validator.validate(configurations);
        // validate the document - uses the default throwing error-handler
    } catch (const std::exception& e) {
        std::cerr << "Validation failed, here is why: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    for (auto& config : configurations) {
        try {
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
