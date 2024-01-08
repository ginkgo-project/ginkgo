// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <tuple>


#include "benchmark/utils/json.hpp"
#include "benchmark/utils/types.hpp"


class BenchmarkOperation {
public:
    virtual ~BenchmarkOperation() = default;

    /**
     * Computes an estimate for the number of FLOPs executed by the operation.
     */
    virtual gko::size_type get_flops() const = 0;

    /**
     * Computes an estimate for the amount of memory accessed by the operation
     * (bytes).
     */
    virtual gko::size_type get_memory() const = 0;

    /**
     * Sets up all necessary data for a following call to
     * BenchmarkOperation::run.
     */
    virtual void prepare() {}

    /**
     * Computes the error between a reference solution and the solution provided
     * by this operation. The first value specifies whether the result is
     * structurally correct, the second value specifies the numerical error.
     */
    virtual std::pair<bool, double> validate() const = 0;

    /**
     * Executes the operation to be benchmarked.
     */
    virtual void run() = 0;


    /**
     * Allows the operation to write arbitrary information to the JSON output.
     */
    virtual void write_stats(json& object) {}
};


using Mtx = gko::matrix::Csr<etype, itype>;


std::unique_ptr<BenchmarkOperation> get_operation(std::string name,
                                                  const Mtx* matrix);
