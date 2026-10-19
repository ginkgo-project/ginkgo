// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_CONFIG_JSON_HPP_
#define GKO_BENCHMARK_UTILS_CONFIG_JSON_HPP_


#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/json.hpp"


/**
 * The parameters of a factory as json. This harness runs on Ginkgo versions
 * without gko::config::write_config, so it records none: always null.
 */
inline json parameters_json(const gko::LinOpFactory&) { return nullptr; }


#endif  // GKO_BENCHMARK_UTILS_CONFIG_JSON_HPP_
