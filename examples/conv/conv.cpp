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
    auto exec = gko::ReferenceExecutor::create();

    // 1D Convolution //

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
    auto host_output = output->clone(exec->get_master());

    std::cout << "Convolution result: ";
    std::cout << std::endl;

    for (gko::size_type i = 0; i < output_length; ++i) {
        std::cout << host_output->at(i, 0) << " ";
    }
    std::cout << std::endl;

    // 2D Convolution //
    // Convolution kernel on the executor
    std::vector<ValueType> kernel_vals_2d_1{1.0, 2.0, 3.0, 4.0, 5.0,
                                            6.0, 7.0, 8.0, 9.0};

    auto kernel_2d_1 = gko::share(
        gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{3, 3}));
    std::copy(kernel_vals_2d_1.begin(), kernel_vals_2d_1.end(),
              kernel_2d_1->get_values());

    std::vector<ValueType> kernel_vals_2d_2{1.0, 2.0, 4.0, 4.0, 8.0,
                                            6.0, 7.0, 8.0, 9.0};

    auto kernel_2d_2 = gko::share(
        gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{3, 3}));
    std::copy(kernel_vals_2d_2.begin(), kernel_vals_2d_2.end(),
              kernel_2d_2->get_values());

    std::vector<std::shared_ptr<const gko::matrix::Dense<ValueType>>>
        kernel_2d = {kernel_2d_1, kernel_2d_2};
    auto conv_op_2d = gko::matrix::Conv2d<ValueType>::create(exec, kernel_2d);


    // Input signal on the executor
    std::vector<ValueType> input_vals_2d{
        1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};

    auto input_2d = gko::share(
        gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{5, 4}));
    std::copy(input_vals_2d.begin(), input_vals_2d.end(),
              input_2d->get_values());

    // Allocate output Dense matrix: floor((N + 2*padding - K) / stride) + 1
    const gko::size_type kernel_rows = kernel_2d.front()->get_size()[0];
    const gko::size_type kernel_cols = kernel_2d.front()->get_size()[1];
    const gko::size_type output_length_2d_rows =
        (input_2d->get_size()[0] + 2 * 0 - kernel_rows) / 1 + 1;
    const gko::size_type output_length_2d_cols =
        (input_2d->get_size()[1] + 2 * 0 - kernel_cols) / 1 + 1;

    std::cout << "Output length (rows): " << output_length_2d_rows << std::endl;
    std::cout << "Output length (cols): " << output_length_2d_cols << std::endl;
    std::vector<std::shared_ptr<gko::matrix::Dense<ValueType>>> outputs_2d;
    for (size_t k = 0; k < kernel_2d.size(); ++k) {
        auto out = gko::matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{output_length_2d_rows, output_length_2d_cols});
        out->fill(0.0);
        outputs_2d.push_back(std::move(out));
    }

    // Apply convolution: conv_op_2d * input_2d -> outputs_2d
    // Convert Dense<ValueType> outputs to LinOp pointers
    std::vector<std::shared_ptr<gko::LinOp>> linop_outputs;
    linop_outputs.reserve(outputs_2d.size());
    for (auto& o : outputs_2d) {
        linop_outputs.push_back(std::static_pointer_cast<gko::LinOp>(o));
    }

    // Apply convolution: conv_op_2d * input_2d -> outputs_2d
    conv_op_2d->apply(input_2d, linop_outputs);

    // Print all outputs
    for (size_t k = 0; k < outputs_2d.size(); ++k) {
        std::cout << "Convolution result (2D, filter " << k << "):\n";
        auto host_output_2d = outputs_2d[k]->clone(exec->get_master());
        for (gko::size_type i = 0; i < output_length_2d_rows; ++i) {
            for (gko::size_type j = 0; j < output_length_2d_cols; ++j) {
                std::cout << host_output_2d->at(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    /*
        // Output the results (2D)
        auto host_output_2d = output_2d->clone(exec->get_master());
        std::cout << "Convolution result (2D): ";
        std::cout << std::endl;
        for (gko::size_type i = 0; i < output_length_2d_rows; ++i) {
            for (gko::size_type j = 0; j < output_length_2d_cols; ++j) {
                std::cout << host_output_2d->at(i, j) << " ";
            }
            std::cout << std::endl;
        }
        //std::cout << std::endl;
        */
}
