// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_UTILS_GENERATOR_HPP
#define GINKGO_BENCHMARK_UTILS_GENERATOR_HPP


#include <string>

#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/stencil_matrix.hpp"
#if GINKGO_BUILD_MPI
#include "benchmark/utils/distributed_helpers.hpp"
#endif


template <typename ValueType = etype, typename IndexType = itype>
struct DefaultSystemGenerator {
    using index_type = IndexType;
    using value_type = ValueType;
    using Vec = vec<ValueType>;

    static std::pair<gko::matrix_data<ValueType, IndexType>, gko::dim<2>>
    generate_matrix_data(const json& config)
    {
        auto [data, size] = [&] {
            if (config["operator"].contains("filename")) {
                std::ifstream in(
                    config["operator"]["filename"].get<std::string>());
                // Returning an empty dim means that there is no specified local
                // size, which is relevant in the distributed case
                return std::make_pair(
                    gko::read_generic_raw<ValueType, IndexType>(in),
                    gko::dim<2>());
            } else if (config["operator"].contains("stencil")) {
                return generate_stencil<ValueType, IndexType>(
                    config["operator"]["stencil"]["name"].get<std::string>(),
                    config["operator"]["stencil"]["size"].get<gko::int64>());
            } else {
                std::cout << config << std::endl;
                throw std::runtime_error(
                    "No known way to generate matrix data found.");
            }
        }();
        data.sort_row_major();
        return {data, size};
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_optimal_format(
        std::shared_ptr<gko::Executor> exec, json& config)
    {
        auto data = generate_matrix_data(config);
        return generate_matrix_with_format(
            std::move(exec),
            config["optimal"]["spmv"]["format"].get<std::string>(), data);
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<gko::Executor> exec, const std::string& format_name,
        const gko::matrix_data<ValueType, itype>& data,
        [[maybe_unused]] const gko::dim<2> size, json* spmv_case = nullptr)
    {
        auto storage_logger = std::make_shared<StorageLogger>();
        if (spmv_case) {
            exec->add_logger(storage_logger);
        }

        auto mtx =
            gko::share(::formats::matrix_factory(format_name, exec, data));

        if (spmv_case) {
            exec->remove_logger(storage_logger);
            storage_logger->write_data(*spmv_case);
        }

        return mtx;
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_default_format(
        std::shared_ptr<gko::Executor> exec,
        const gko::matrix_data<ValueType, itype>& data,
        const gko::dim<2> local_size)
    {
        return generate_matrix_with_format(std::move(exec), "coo", data,
                                           local_size);
    }

    static gko::dim<2> create_default_local_size(gko::dim<2> global_size)
    {
        return global_size;
    }

    static std::unique_ptr<Vec> create_multi_vector(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> global_size,
        [[maybe_unused]] gko::dim<2> local_size, ValueType value)
    {
        auto res = Vec::create(exec);
        res->read(gko::matrix_data<ValueType, itype>(global_size, value));
        return res;
    }

    static std::unique_ptr<Vec> create_multi_vector_strided(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> global_size,
        [[maybe_unused]] gko::dim<2> local_size, gko::size_type stride)
    {
        auto res = Vec::create(exec, global_size, stride);
        return res;
    }

    // creates a random multi_vector
    static std::unique_ptr<Vec> create_multi_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> global_size,
        [[maybe_unused]] gko::dim<2> local_size)
    {
        auto res = Vec::create(exec);
        res->read(gko::matrix_data<ValueType, itype>(
            global_size,
            std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1.0,
                                                                           1.0),
            get_engine()));
        return res;
    }

    static std::unique_ptr<Vec> initialize(
        std::initializer_list<ValueType> il,
        std::shared_ptr<const gko::Executor> exec)
    {
        return gko::initialize<Vec>(std::move(il), std::move(exec));
    }
};


#if GINKGO_BUILD_MPI


template <typename LocalGeneratorType>
struct DistributedDefaultSystemGenerator {
    using LocalGenerator = LocalGeneratorType;

    using index_type = global_itype;
    using local_index_type = typename LocalGenerator::index_type;
    using value_type = typename LocalGenerator::value_type;

    using Mtx = dist_mtx<value_type, local_index_type, index_type>;
    using Vec = dist_vec<value_type>;

    std::pair<gko::matrix_data<value_type, index_type>, gko::dim<2>>
    generate_matrix_data(const json& config) const
    {
        auto [data, local_size] = [&] {
            if (config["operator"].contains("filename")) {
                std::ifstream in(
                    config["operator"]["filename"].get<std::string>());
                // Returning an empty dim means that no local size is specified,
                // and thus the partition has to be deduced from the global size
                return std::make_pair(
                    gko::read_generic_raw<value_type, index_type>(in),
                    gko::dim<2>());
            } else if (config["operator"].contains("stencil")) {
                auto& stencil = config["operator"]["stencil"];
                return generate_stencil<value_type, index_type>(
                    stencil["name"].get<std::string>(), comm,
                    stencil["local_size"].get<gko::int64>(),
                    stencil.value("process_grid", std::vector<gko::int64>()));
            } else {
                throw std::runtime_error(
                    "No known way to generate matrix data found.");
            }
        }();
        data.sort_row_major();
        return {data, local_size};
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<gko::Executor> exec, const std::string& format_name,
        const gko::matrix_data<value_type, index_type>& data,
        const gko::dim<2> local_size, json* spmv_case = nullptr) const
    {
        auto part = gko::share(
            local_size
                ? gko::experimental::distributed::
                      build_partition_from_local_size<itype, global_itype>(
                          exec, comm, local_size[0])
                : gko::experimental::distributed::Partition<itype,
                                                            global_itype>::
                      build_from_global_size_uniform(exec, comm.size(),
                                                     data.size[0]));
        auto formats = split(format_name, '-');
        if (formats.size() != 2) {
            throw std::runtime_error{"Invalid distributed format specifier " +
                                     format_name};
        }

        auto local_mat = formats::matrix_type_factory.at(formats[0])(exec);
        auto non_local_mat = formats::matrix_type_factory.at(formats[1])(exec);

        auto storage_logger = std::make_shared<StorageLogger>();
        if (spmv_case) {
            exec->add_logger(storage_logger);
        }

        auto dist_mat = dist_mtx<etype, itype, global_itype>::create(
            exec, comm, local_mat, non_local_mat);
        dist_mat->read_distributed(data, part);

        if (spmv_case) {
            exec->remove_logger(storage_logger);
            storage_logger->write_data(comm, *spmv_case);
        }

        return dist_mat;
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_default_format(
        std::shared_ptr<gko::Executor> exec,
        const gko::matrix_data<value_type, global_itype>& data,
        const gko::dim<2> local_size) const
    {
        return generate_matrix_with_format(std::move(exec), "coo-coo", data,
                                           local_size);
    }

    gko::dim<2> create_default_local_size(gko::dim<2> global_size) const
    {
        // This computes Partition::build_from_global_size_uniform manually,
        // since otherwise an executor would be necessary
        const auto num_parts = comm.size();
        const auto size_per_part = global_size[0] / num_parts;
        const auto rest = global_size[0] - (num_parts * size_per_part);
        return gko::dim<2>{size_per_part + (comm.rank() < rest ? 1 : 0),
                           global_size[1]};
    }

    std::unique_ptr<Vec> create_multi_vector(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> global_size,
        gko::dim<2> local_size, value_type value) const
    {
        return Vec::create(exec, comm, global_size,
                           local_generator.create_multi_vector(
                               exec, local_size, local_size, value));
    }

    std::unique_ptr<Vec> create_multi_vector_strided(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> global_size,
        gko::dim<2> local_size, gko::size_type stride) const
    {
        return Vec::create(exec, comm, global_size,
                           local_generator.create_multi_vector_strided(
                               exec, local_size, local_size, stride));
    }

    // creates a random multi_vector
    std::unique_ptr<Vec> create_multi_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> global_size,
        gko::dim<2> local_size) const
    {
        return Vec::create(exec, comm, global_size,
                           local_generator.create_multi_vector_random(
                               exec, local_size, local_size));
    }

    std::unique_ptr<Vec> initialize(
        std::initializer_list<value_type> il,
        std::shared_ptr<const gko::Executor> exec) const
    {
        auto local = gko::initialize<typename LocalGenerator::Vec>(
            std::move(il), std::move(exec));
        auto global_rows = local->get_size()[0];
        comm.all_reduce(gko::ReferenceExecutor::create(), &global_rows, 1,
                        MPI_SUM);
        return Vec::create(exec, comm,
                           gko::dim<2>{global_rows, local->get_size()[1]},
                           std::move(local));
    }

    gko::experimental::mpi::communicator comm;
    LocalGenerator local_generator{};
};


#endif
#endif  // GINKGO_BENCHMARK_UTILS_GENERATOR_HPP
