// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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

    static gko::matrix_data<ValueType, IndexType> generate_matrix_data(
        const json& config)
    {
        gko::matrix_data<ValueType, IndexType> data;
        if (config.contains("filename")) {
            std::ifstream in(config["filename"].get<std::string>());
            data = gko::read_generic_raw<ValueType, IndexType>(in);
        } else if (config.contains("stencil")) {
            data = generate_stencil<ValueType, IndexType>(
                config["stencil"].get<std::string>(),
                config["size"].get<gko::int64>());
        } else {
            throw std::runtime_error(
                "No known way to generate matrix data found.");
        }
        data.sort_row_major();
        return data;
    }

    static std::string get_example_config()
    {
        return json::
            parse(R"([{"filename": "my_file.mtx"},{"filename": "my_file2.mtx"},{"size": 100, "stencil": "7pt"}])")
                .dump(4);
    }

    static bool validate_config(const json& test_case)
    {
        return ((test_case.contains("size") && test_case.contains("stencil") &&
                 test_case["size"].is_number_integer() &&
                 test_case["stencil"].is_string()) ||
                (test_case.contains("filename") &&
                 test_case["filename"].is_string()));
    }

    static std::string describe_config(const json& config)
    {
        if (config.contains("filename")) {
            return config["filename"].get<std::string>();
        } else if (config.contains("stencil")) {
            std::stringstream ss;
            ss << "stencil(" << config["size"].get<gko::int64>() << ", "
               << config["stencil"].get<std::string>() << ")";
            return ss.str();
        } else {
            throw std::runtime_error("No known way to describe config.");
        }
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_optimal_format(
        std::shared_ptr<gko::Executor> exec, json& config)
    {
        auto data = generate_matrix_data(config);
        return generate_matrix_with_format(
            std::move(exec), config["optimal"]["spmv"].get<std::string>(),
            data);
    }

    static std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<gko::Executor> exec, const std::string& format_name,
        const gko::matrix_data<ValueType, itype>& data,
        json* spmv_case = nullptr)
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
        const gko::matrix_data<ValueType, itype>& data)
    {
        return generate_matrix_with_format(std::move(exec), "coo", data);
    }

    static std::unique_ptr<Vec> create_multi_vector(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        ValueType value)
    {
        auto res = Vec::create(exec);
        res->read(gko::matrix_data<ValueType, itype>(size, value));
        return res;
    }

    static std::unique_ptr<Vec> create_multi_vector_strided(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        gko::size_type stride)
    {
        auto res = Vec::create(exec, size, stride);
        return res;
    }

    // creates a random multi_vector
    static std::unique_ptr<Vec> create_multi_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
    {
        auto res = Vec::create(exec);
        res->read(gko::matrix_data<ValueType, itype>(
            size,
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

    gko::matrix_data<value_type, index_type> generate_matrix_data(
        const json& config) const
    {
        gko::matrix_data<value_type, index_type> data;
        if (config.contains("filename")) {
            std::ifstream in(config["filename"].get<std::string>());
            data = gko::read_generic_raw<value_type, index_type>(in);
        } else if (config.contains("stencil")) {
            auto local_size = static_cast<global_itype>(
                config["size"].get<gko::int64>() / comm.size());
            data = generate_stencil<value_type, index_type>(
                config["stencil"].get<std::string>(), comm, local_size,
                config["comm_pattern"].get<std::string>() ==
                    std::string("optimal"));
        } else {
            throw std::runtime_error(
                "No known way to generate matrix data found.");
        }
        data.sort_row_major();
        return data;
    }

    static std::string get_example_config()
    {
        return json::
            parse(R"([{"size": 100, "stencil": "7pt", "comm_pattern": "stencil"}, {"filename": "my_file.mtx"}])")
                .dump(4);
    }

    static bool validate_config(const json& test_case)
    {
        return ((test_case.contains("size") && test_case.contains("stencil") &&
                 test_case.contains("comm_pattern") &&
                 test_case["size"].is_number_integer() &&
                 test_case["stencil"].is_string() &&
                 test_case["comm_pattern"].is_string()) ||
                (test_case.contains("filename") &&
                 test_case["filename"].is_string()));
    }

    static std::string describe_config(const json& config)
    {
        if (config.contains("filename")) {
            return config["filename"].get<std::string>();
        } else if (config.contains("stencil")) {
            std::stringstream ss;
            ss << "stencil(" << config["size"].get<gko::int64>() << ", "
               << config["stencil"].get<std::string>() << ", "
               << config["comm_pattern"].get<std::string>() << ")";
            return ss.str();
        } else {
            throw std::runtime_error("No known way to describe config.");
        }
    }

    std::shared_ptr<gko::LinOp> generate_matrix_with_format(
        std::shared_ptr<gko::Executor> exec, const std::string& format_name,
        const gko::matrix_data<value_type, index_type>& data,
        json* spmv_case = nullptr) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, global_itype>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<global_itype>(data.size[0]));
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
        const gko::matrix_data<value_type, global_itype>& data) const
    {
        return generate_matrix_with_format(std::move(exec), "coo-coo", data);
    }

    std::unique_ptr<Vec> create_multi_vector(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        value_type value) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, global_itype>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<global_itype>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator.create_multi_vector(
                exec,
                gko::dim<2>{static_cast<gko::size_type>(
                                part->get_part_size(comm.rank())),
                            size[1]},
                value));
    }

    std::unique_ptr<Vec> create_multi_vector_strided(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        gko::size_type stride) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, global_itype>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<global_itype>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator.create_multi_vector_strided(
                exec,
                gko::dim<2>{static_cast<gko::size_type>(
                                part->get_part_size(comm.rank())),
                            size[1]},
                stride));
    }

    // creates a random multi_vector
    std::unique_ptr<Vec> create_multi_vector_random(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size) const
    {
        auto part = gko::experimental::distributed::
            Partition<itype, global_itype>::build_from_global_size_uniform(
                exec, comm.size(), static_cast<global_itype>(size[0]));
        return Vec::create(
            exec, comm, size,
            local_generator.create_multi_vector_random(
                exec, gko::dim<2>{static_cast<gko::size_type>(
                                      part->get_part_size(comm.rank())),
                                  size[1]}));
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
