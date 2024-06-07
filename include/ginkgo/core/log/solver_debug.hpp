// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_LOG_SOLVER_DEBUG_HPP_
#define GKO_PUBLIC_CORE_LOG_SOLVER_DEBUG_HPP_


#include <iosfwd>
#include <memory>


#include <ginkgo/core/log/logger.hpp>


namespace gko {
namespace log {


/**
 * This Logger prints the value of all scalar values stored internally by the
 * solver after each iteration. If the solver is applied to multiple right-hand
 * sides, only the first right-hand side gets printed.
 */
class SolverDebug : public Logger {
public:
    /**
     * Creates a logger printing the value for all scalar values in the solver
     * after each iteration in an ASCII table.
     *
     * @param output  the stream to write the output to.
     * @param precision  the number of digits of precision to print
     * @param column_width  the number of characters an output column is wide
     */
    static std::shared_ptr<SolverDebug> create_scalar_table(
        std::ostream& output, int precision = 6, int column_width = 12);


    /**
     * Creates a logger printing the value for all scalar values in the solver
     * after each iteration in a CSV table.
     *
     * @param output  the stream to write the output to.
     * @param precision  the number of digits of precision to print
     * @param column_width  the number of characters an output column is wide
     */
    static std::shared_ptr<SolverDebug> create_scalar_csv(std::ostream& output,
                                                          int precision = 6,
                                                          char separator = ',');


    /**
     * Creates a logger storing all vectors and scalar values in the solver
     * after each iteration on disk.
     *
     * @param output  the path and file name prefix used to generate the output
     *                file names.
     * @param precision  the number of digits of precision to print when
     *                   outputting matrices in text format
     * @param binary  if true, write data in Ginkgo's own binary format
     *                (lossless), if false write data in the MatrixMarket format
     *                (potentially lossy)
     */
    static std::shared_ptr<SolverDebug> create_vector_storage(
        std::string output_file_prefix = "solver_", bool binary = false);
};


}  // namespace log
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_LOG_SOLVER_DEBUG_HPP_
