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

    // Executors
    auto cpu = gko::ReferenceExecutor::create();
    //auto exec = gko::CudaExecutor::create(0, cpu);  // GPU executor
    auto exec = gko::ReferenceExecutor::create();

    // ==============================================================
    // 1D Convolution
    // ==============================================================

    std::vector<ValueType> kernel_vals{1.0, 2.0, 3.0};
    gko::array<ValueType> kernel_array(exec, kernel_vals.begin(),
                                       kernel_vals.end());
    auto conv_op = gko::matrix::Conv<ValueType>::create(exec, kernel_array);

    // Input
    auto input = gko::initialize<Vec>({4.0, 5.0, 6.0, 7.0}, exec);
    const gko::size_type output_length =
        (input->get_size()[0] + 2 * 0 - kernel_vals.size()) / 1 + 1;

    auto output = Vec::create(exec, gko::dim<2>{output_length, 1});
    output->fill(0.0);

    conv_op->apply(input, output);

    auto host_output = output->clone(cpu);

    std::cout << "Convolution result (1D): ";
    for (gko::size_type i = 0; i < output_length; ++i) {
        std::cout << host_output->at(i, 0) << " ";
    }
    std::cout << "\n" << std::endl;


    // ==============================================================
    // 2D Convolution
    // ==============================================================

    // --- Kernel 1 and 2 (create on host first)
    std::vector<ValueType> kernel_vals_2d_1{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0};

    std::vector<ValueType> kernel_vals_2d_2{
        1.0, 2.0, 4.0,
        4.0, 8.0, 6.0,
        7.0, 8.0, 9.0};

    auto kernel_2d_1_host =
        gko::matrix::Dense<ValueType>::create(cpu, gko::dim<2>{3, 3});
    std::copy(kernel_vals_2d_1.begin(), kernel_vals_2d_1.end(),
              kernel_2d_1_host->get_values());

    auto kernel_2d_2_host =
        gko::matrix::Dense<ValueType>::create(cpu, gko::dim<2>{3, 3});
    std::copy(kernel_vals_2d_2.begin(), kernel_vals_2d_2.end(),
              kernel_2d_2_host->get_values());

    // Copy to device + convert to shared_ptr<const Dense<ValueType>>
    auto kernel_2d_1 = gko::share(
        gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{3, 3}));
    auto kernel_2d_2 = gko::share(
        gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{3, 3}));
    kernel_2d_1->copy_from(kernel_2d_1_host);
    kernel_2d_2->copy_from(kernel_2d_2_host);

    std::vector<std::shared_ptr<const gko::matrix::Dense<ValueType>>> kernels_2d = {
        kernel_2d_1, kernel_2d_2};

    auto conv_op_2d = gko::matrix::Conv2d<ValueType>::create(exec, kernels_2d);


    // --- Input image (5x4)
    std::vector<ValueType> input_vals_2d{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0};

    auto input_2d_host =
        gko::matrix::Dense<ValueType>::create(cpu, gko::dim<2>{5, 4});
    std::copy(input_vals_2d.begin(), input_vals_2d.end(),
              input_2d_host->get_values());

    auto input_2d = gko::share(
        gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{5, 4}));
    input_2d->copy_from(input_2d_host);


    // --- Output
    const gko::size_type kernel_rows = kernels_2d.front()->get_size()[0];
    const gko::size_type kernel_cols = kernels_2d.front()->get_size()[1];
    const gko::size_type output_rows =
        (input_2d->get_size()[0] + 2 * 0 - kernel_rows) / 1 + 1;
    const gko::size_type output_cols =
        (input_2d->get_size()[1] + 2 * 0 - kernel_cols) / 1 + 1;

    std::vector<std::shared_ptr<gko::matrix::Dense<ValueType>>> outputs_2d;
    for (size_t k = 0; k < kernels_2d.size(); ++k) {
        auto out = gko::share(
            gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{output_rows, output_cols}));
        out->fill(0.0);
        outputs_2d.push_back(out);
    }

    // --- Convert outputs to LinOp
    std::vector<std::shared_ptr<gko::LinOp>> linop_outputs;
    for (auto& o : outputs_2d) {
        linop_outputs.push_back(o);
    }

    // --- Apply convolution
// --- Apply convolution
conv_op_2d->apply(input_2d, linop_outputs);

// --- Wait for GPU to finish before copying back
exec->synchronize();

    // --- Print results
    for (size_t k = 0; k < outputs_2d.size(); ++k) {
        std::cout << "Convolution result (2D, filter " << k << "):\n";
        auto host_output_2d = outputs_2d[k]->clone(cpu);
        for (gko::size_type i = 0; i < output_rows; ++i) {
            for (gko::size_type j = 0; j < output_cols; ++j) {
                std::cout << host_output_2d->at(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}

