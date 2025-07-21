// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;

#include "cerebras_interface.hpp"
#include "python_handler.hpp"

#ifndef M
#define M 1000
#endif
#ifndef G
#define G 100
#endif


void print_matrix(std::vector<float>& matrix)
{
    int size = (int)std::sqrt((double)matrix.size());
    std::cout << "MATRIX SIZE = " << size << std::endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            std::cout << matrix[i * size + j] << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(void)
{
    // true is here to load onto simulator, not real device
    CerebrasInterface cerebras(std::string("cerebras_python_interface"), true);

    auto matrix = std::vector<float>();
    std::random_device rd;
    std::default_random_engine re(rd());
    std::uniform_real_distribution<float> unif(-10., 10.);
    for (int i = 0; i < M * M; i++) {
        // matrix.emplace_back(unif(re));
        matrix.emplace_back(-1.);
    }
    for (int i = 0; i < M; i++) {
        double s = std::abs(matrix[i * M + i]);
        for (int j = M * i; j < ((i + 1) * M) - 1; j++) {
            s += std::abs(matrix[j]);
        }
        matrix[i * M + i] = s;
    }

    auto start = Clock::now();
    cerebras.copy_h2d("A", matrix, 0, 0, G, G, (M / G) * (M / G), false, false);
    auto end = Clock::now();
    Duration d = end - start;
    std::cout << "[DEBUG] H2D TIME = " << d.count() << "[mms]" << std::endl;

    start = Clock::now();
    cerebras.call_func("start");
    end = Clock::now();
    d = end - start;
    std::cout << "[DEBUG] COMP. TIME = " << d.count() << "[mms]" << std::endl;

    //    start = Clock::now();
    //    cerebras.copy_d2h(
    //        "A",
    //        matrix,
    //        0, 0,
    //        G, G,
    //        (M / G) * (M / G),
    //        false,
    //        false
    //    );
    //    end = Clock::now();
    //    d = end - start;
    //    std::cout << "[DEBUG] D2H TIME = " << d.count() << "[mms]" <<
    //    std::endl;

    return 0;
}
