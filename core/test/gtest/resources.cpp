// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <regex>
#include <sstream>


#include "core/test/gtest/resources.hpp"


#ifdef GKO_COMPILING_OMP
#include <omp.h>
#endif


#ifdef GKO_COMPILING_CUDA
#include "cuda/base/device.hpp"
#endif


#ifdef GKO_COMPILING_HIP
#include "hip/base/device.hpp"
#endif


#if GKO_COMPILING_DPCPP
#include "dpcpp/base/device.hpp"
#endif


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>


struct ctest_resource {
    int id;
    int slots;
};


char* get_ctest_group(std::string resource_type, int group_id)
{
    std::transform(resource_type.begin(), resource_type.end(),
                   resource_type.begin(),
                   [](auto c) { return std::toupper(c); });
    std::string rs_group_env = "CTEST_RESOURCE_GROUP_" +
                               std::to_string(group_id) + "_" + resource_type;
    return std::getenv(rs_group_env.c_str());
}


ctest_resource parse_ctest_resources(std::string resource)
{
    std::regex re(R"(id\:(\d+),slots\:(\d+))");
    std::smatch match;

    if (!std::regex_match(resource, match, re)) {
        GKO_INVALID_STATE("Can't parse ctest_resource string: " + resource);
    }

    return ctest_resource{std::stoi(match[1]), std::stoi(match[2])};
}


ResourceEnvironment::ResourceEnvironment(int rank, int size)
{
#if GINKGO_BUILD_MPI
    if (size > 1) {
        cuda_device_id = gko::experimental::mpi::map_rank_to_device_id(
            MPI_COMM_WORLD, std::max(gko::CudaExecutor::get_num_devices(), 1));
        hip_device_id = gko::experimental::mpi::map_rank_to_device_id(
            MPI_COMM_WORLD, std::max(gko::HipExecutor::get_num_devices(), 1));
        sycl_device_id = gko::experimental::mpi::map_rank_to_device_id(
            MPI_COMM_WORLD,
            std::max(gko::DpcppExecutor::get_num_devices("gpu"), 1));
    }
#endif

    auto rs_count_env = std::getenv("CTEST_RESOURCE_GROUP_COUNT");
    auto rs_count = rs_count_env ? std::stoi(rs_count_env) : 0;
    if (rs_count == 0) {
        if (rank == 0) {
            std::cerr << "Running without CTest ctest_resource configuration"
                      << std::endl;
        }
        return;
    }
    if (rs_count != size) {
        GKO_INVALID_STATE("Invalid resource group count: " +
                          std::to_string(rs_count));
    }

    // parse CTest ctest_resource group descriptions
    // OpenMP CPU threads
    if (auto rs_omp_env = get_ctest_group("cpu", rank)) {
        auto resource = parse_ctest_resources(rs_omp_env);
        omp_threads = resource.slots;
    }
    // CUDA GPUs
    if (auto rs_cuda_env = get_ctest_group("cudagpu", rank)) {
        auto resource = parse_ctest_resources(rs_cuda_env);
        cuda_device_id = resource.id;
    }
    // HIP GPUs
    if (auto rs_hip_env = get_ctest_group("hipgpu", rank)) {
        auto resource = parse_ctest_resources(rs_hip_env);
        hip_device_id = resource.id;
    }
    // SYCL GPUs (no other devices!)
    if (auto rs_sycl_env = get_ctest_group("sycl", rank)) {
        auto resource = parse_ctest_resources(rs_sycl_env);
        sycl_device_id = resource.id;
    }
}
