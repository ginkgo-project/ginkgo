// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_JSON_HPP_
#define GKO_BENCHMARK_UTILS_JSON_HPP_


#include <nlohmann/json-schema.hpp>
#include <nlohmann/json.hpp>


using json = nlohmann::ordered_json;


namespace json_schema = nlohmann::json_schema;


static void json_loader(const nlohmann::json_uri& uri,
                        nlohmann::basic_json<>& schema)
{
    std::string filename = GKO_ROOT "/benchmark/schema/" + uri.path();
    std::ifstream lf(filename);
    if (!lf.good())
        throw std::invalid_argument("could not open " + uri.url() +
                                    " tried with " + filename);
    try {
        lf >> schema;
    } catch (const std::exception& e) {
        throw e;
    }
}


#endif  // GKO_BENCHMARK_UTILS_JSON_HPP_
