// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iostream>
#include <memory>
#include <vector>

#include <ginkgo/ginkgo.hpp>

int main()
{
    using ValueType = double;
    using Vec = gko::matrix::Dense<ValueType>;

    // Executor setup
    // auto exec = gko::OmpExecutor::create();
    auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());


    // Convolution kernel (length K) as a gko::array on the executor
    std::vector<ValueType> kernel_vals{1.0, 2.0, 3.0};
    gko::array<ValueType> kernel_array(exec, kernel_vals.begin(),
                                       kernel_vals.end());
    auto conv_op = gko::matrix::Conv<ValueType>::create(exec, kernel_array);

    // Input signal (length N) as a Dense vector
    auto input = gko::initialize<Vec>({4.0, 5.0, 6.0, 7.0}, exec);

    // Allocate output Dense vector: floor((N + 2*padding - K) / stride) + 1
    // elements
    const gko::size_type output_length =
        (input->get_size()[0] + 2 * 0 - kernel_vals.size()) / 1 + 1;
    std::cout << "Output length: " << output_length << std::endl;
    auto output = Vec::create(exec, gko::dim<2>{output_length, 1});
    output->fill(0.0);

    // Apply convolution: conv_op * input -> output
    conv_op->apply(gko::lend(input), gko::lend(output));

    // Output the results
    std::cout << "Convolution result: ";
    for (gko::size_type i = 0; i < output_length; ++i) {
        std::cout << output->at(i, 0) << " ";
    }
    std::cout << std::endl;
}
