// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_SPMV_VALIDATION_HPP_
#define GKO_BENCHMARK_UTILS_SPMV_VALIDATION_HPP_


#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <iostream>


#include <rapidjson/document.h>


std::string example_config = R"(
  [
    {"filename": "my_file.mtx"},
    {"filename": "my_file2.mtx"},
    {"size": 100, "stencil": "7pt"},
  ]
)";


/**
 * Function which outputs the input format for benchmarks similar to the spmv.
 */
[[noreturn]] void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << example_config << std::endl;
    std::exit(1);
}


/**
 * Validates whether the input format is correct for spmv-like benchmarks.
 *
 * @param value  the JSON value to test.
 */
void validate_option_object(const rapidjson::Value& value)
{
    if (!value.IsObject() ||
        !((value.HasMember("size") && value.HasMember("stencil") &&
           value["size"].IsInt64() && value["stencil"].IsString()) ||
          (value.HasMember("filename") && value["filename"].IsString()))) {
        print_config_error_and_exit();
    }
}


#endif  // GKO_BENCHMARK_UTILS_SPMV_VALIDATION_HPP_
