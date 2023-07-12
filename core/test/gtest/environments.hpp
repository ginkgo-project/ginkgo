#ifndef GINKGO_ENVIRONMENTS_HPP
#define GINKGO_ENVIRONMENTS_HPP

#include <algorithm>
#include <regex>


#include <ginkgo/core/base/exception_helpers.hpp>


std::vector<std::string> split(const std::string& s, char delimiter = ',')
{
    std::istringstream iss(s);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}


struct resource {
    int id;
    int slots;
};

resource parse_single_resource(const std::string& resource_string)
{
    std::regex re(R"(id\:(\d+),slots\:(\d+))");
    std::smatch match;

    if (!std::regex_match(resource_string, match, re)) {
        GKO_INVALID_STATE("Can't parse resource string: " + resource_string);
    }

    return resource{std::stoi(match[1]), std::stoi(match[2])};
}

std::vector<resource> parse_all_resources(const std::string& resource_string)
{
    auto resource_strings = split(resource_string, ';');

    std::vector<resource> resources;
    for (const auto& rs : resource_strings) {
        resources.push_back(parse_single_resource(rs));
    }
    return resources;
}


std::vector<resource> get_ctest_resources()
{
    auto rs_count_env = std::getenv("CTEST_RESOURCE_GROUP_COUNT");

    if (!rs_count_env) {
        return {{0, 1}};
    }

    auto rs_count = std::stoi(rs_count_env);

    if (rs_count > 1) {
        GKO_INVALID_STATE("Can handle only one resource group.");
    }

    std::string rs_type = std::getenv("CTEST_RESOURCE_GROUP_0");
    std::transform(rs_type.begin(), rs_type.end(), rs_type.begin(),
                   [](auto c) { return std::toupper(c); });
    std::string rs_env =
        std::getenv(std::string("CTEST_RESOURCE_GROUP_0_" + rs_type).c_str());
    std::cerr << rs_env << std::endl;
    return parse_all_resources(rs_env);
}


class ResourceEnvironment : public ::testing::Environment {
public:
    explicit ResourceEnvironment(resource rs_) : ::testing::Environment()
    {
        rs = rs_;
    }

    static resource rs;
};


#ifdef GKO_COMPILING_OMP

#include <omp.h>

class OmpEnvironment : public ::testing::Environment {
public:
    void SetUp() override
    {
        omp_set_num_threads(ResourceEnvironment::rs.slots);
    }
};

#else


class OmpEnvironment : public ::testing::Environment {};

#endif


#ifdef GKO_COMPILING_CUDA

#include "cuda/base/device.hpp"

class CudaEnvironment : public ::testing::Environment {
public:
    void TearDown() override
    {
        gko::kernels::cuda::reset_device(ResourceEnvironment::rs.id);
    }
};

#else

class CudaEnvironment : public ::testing::Environment {};

#endif


#ifdef GKO_COMPILING_HIP

#include "hip/base/device.hpp"

class HipEnvironment : public ::testing::Environment {
public:
    void TearDown() override
    {
        gko::kernels::hip::reset_device(ResourceEnvironment::rs.id);
    }
};

#else

class HipEnvironment : public ::testing::Environment {};

#endif


#endif  // GINKGO_ENVIRONMENTS_HPP
