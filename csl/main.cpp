// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

#include "cerebras_interface.hpp"
#include "python_handler.hpp"


int csl_test_kernel(void)
{
    // true is here to load onto simulator, not real device
    CerebrasInterface cerebras(true);

    auto matrix = std::vector<float>();
    for (int i = 0; i < M * M; i++) {
        matrix.emplace_back(-1.);
    }

    cerebras.copy_h2d("A", matrix, 0, 0, G, G, (M / G) * (M / G), false, false);
    cerebras.call_func("start");
    cerebras.copy_h2d("A", matrix, 0, 0, G, G, (M / G) * (M / G), false, false);


    return 0;
}
