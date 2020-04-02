/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "mfem.hpp"
#include "mfem_wrapper.hpp"

#include <ginkgo/ginkgo.hpp>

// ------ Begin Ginkgo custom logger definition and auxiliary functions ------
// Adapted from the Ginkgo `custom-logger` example, see
//   `ginkgo/examples/custom-logger/custom-logger.cpp`
//
// Utility function which gets the scalar value of a Ginkgo gko::matrix::Dense
// matrix representing the norm of a vector.
template <typename ValueType>
double get_norm(const gko::matrix::Dense<ValueType> *norm)
{
    // Put the value on CPU thanks to the master executor
    auto cpu_norm = clone(norm->get_executor()->get_master(), norm);
    // Return the scalar value contained at position (0, 0)
    return cpu_norm->at(0, 0);
}

template <typename ValueType>
double compute_norm(const gko::matrix::Dense<ValueType> *b)
{
    // Get the executor of the vector
    auto exec = b->get_executor();
    // Initialize a result scalar containing the value 0.0.
    auto b_norm = gko::initialize<gko::matrix::Dense<ValueType>>({0.0}, exec);
    // Use the dense `compute_norm2` function to compute the norm.
    b->compute_norm2(lend(b_norm));
    // Use the other utility function to return the norm contained in `b_norm``
    return get_norm(lend(b_norm));
}

template <typename ValueType = double>
struct ResidualLogger : gko::log::Logger {
    // Output the logger's data in a table format
    void write() const
    {
        // Print a header for the table
        mfem::out << "Iteration log with real residual norms:" << std::endl;
        mfem::out << '|' << std::setw(10) << "Iteration" << '|' << std::setw(25)
                  << "Real Residual Norm" << '|' << std::endl;
        // Print a separation line. Note that for creating `10` characters
        // `std::setw()` should be set to `11`.
        mfem::out << '|' << std::setfill('-') << std::setw(11) << '|'
                  << std::setw(26) << '|' << std::setfill(' ') << std::endl;
        // Print the data one by one in the form
        mfem::out << std::scientific;
        for (std::size_t i = 0; i < iterations.size(); i++) {
            mfem::out << '|' << std::setw(10) << iterations[i] << '|'
                      << std::setw(25) << real_norms[i] << '|' << std::endl;
        }
        // std::defaultfloat could be used here but some compilers do not
        // support it properly, e.g. the Intel compiler
        mfem::out.unsetf(std::ios_base::floatfield);
        // Print a separation line
        mfem::out << '|' << std::setfill('-') << std::setw(11) << '|'
                  << std::setw(26) << '|' << std::setfill(' ') << std::endl;
    }

    using gko_dense = gko::matrix::Dense<ValueType>;

    // Customize the logging hook which is called every time an iteration is
    // completed
    void on_iteration_complete(const gko::LinOp *,
                               const gko::size_type &iteration,
                               const gko::LinOp *residual,
                               const gko::LinOp *solution,
                               const gko::LinOp *residual_norm) const override
    {
        // If the solver shares a residual norm, log its value
        if (residual_norm) {
            auto dense_norm = gko::as<gko_dense>(residual_norm);
            // Add the norm to the `recurrent_norms` vector
            recurrent_norms.push_back(get_norm(dense_norm));
            // Otherwise, use the recurrent residual vector
        } else {
            auto dense_residual = gko::as<gko_dense>(residual);
            // Compute the residual vector's norm
            auto norm = compute_norm(gko::lend(dense_residual));
            // Add the computed norm to the `recurrent_norms` vector
            recurrent_norms.push_back(norm);
        }

        // If the solver shares the current solution vector
        if (solution) {
            // Store the matrix's executor
            auto exec = matrix->get_executor();
            // Create a scalar containing the value 1.0
            auto one = gko::initialize<gko_dense>({1.0}, exec);
            // Create a scalar containing the value -1.0
            auto neg_one = gko::initialize<gko_dense>({-1.0}, exec);
            // Instantiate a temporary result variable
            auto res = gko::clone(gko::as<MFEMVectorWrapper>(b));
            // Compute the real residual vector by calling apply on the system
            // matrix
            matrix->apply(gko::lend(one),
                          gko::lend(gko::as<MFEMVectorWrapper>(solution)),
                          gko::lend(neg_one), gko::lend(res));

            // Compute the norm of the residual vector and add it to the
            // `real_norms` vector
            real_norms.push_back(
                mfem::GinkgoWrappers::compute_norm(gko::lend(res)));
        } else {
            // Add to the `real_norms` vector the value -1.0 if it could not be
            // computed
            real_norms.push_back(-1.0);
        }

        // Add the current iteration number to the `iterations` vector
        iterations.push_back(iteration);

        // Add the current iteration number to the `iterations` vector
        std::cout << "Iteration  " << iteration << " : residual norm "
                  << recurrent_norms[iteration] << ", real residual norm "
                  << real_norms[iteration] << std::endl;
    }

    // Construct the logger and store the system matrix and b vectors
    ResidualLogger(std::shared_ptr<const gko::Executor> exec,
                   const gko::LinOp *matrix, const MFEMVectorWrapper *b)
        : gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask),
          matrix{matrix},
          b{b}
    {}

private:
    // Pointer to the system matrix
    const gko::LinOp *matrix;
    // Pointer to the right hand sides
    const gko::matrix::Dense<ValueType> *b;
    // Vector which stores all the recurrent residual norms
    mutable std::vector<ValueType> recurrent_norms{};
    // Vector which stores all the real residual norms
    mutable std::vector<ValueType> real_norms{};
    // Vector which stores all the iteration numbers
    mutable std::vector<std::size_t> iterations{};
};
