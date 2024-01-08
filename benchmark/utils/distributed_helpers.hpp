// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_UTILS_DISTRIBUTED_HELPERS_HPP
#define GINKGO_BENCHMARK_UTILS_DISTRIBUTED_HELPERS_HPP


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/stencil_matrix.hpp"


using global_itype = gko::int64;


template <typename ValueType>
using dist_vec = gko::experimental::distributed::Vector<ValueType>;
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
using dist_mtx =
    gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                           GlobalIndexType>;


std::string broadcast_json_input(std::istream& is,
                                 gko::experimental::mpi::communicator comm)
{
    auto exec = gko::ReferenceExecutor::create();

    std::string json_input;
    if (comm.rank() == 0) {
        std::string line;
        while (is >> line) {
            json_input += line;
        }
    }

    auto input_size = json_input.size();
    comm.broadcast(exec->get_master(), &input_size, 1, 0);
    json_input.resize(input_size);
    comm.broadcast(exec->get_master(), &json_input[0],
                   static_cast<int>(input_size), 0);

    return json_input;
}


#endif  // GINKGO_BENCHMARK_UTILS_DISTRIBUTED_HELPERS_HPP
