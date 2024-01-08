// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <cmath>
#include <exception>
#include <iostream>


#include <ginkgo/core/base/executor.hpp>


#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// See en.wikipedia.org/wiki/Five-number_summary
// Quartile computation uses Method 3 from en.wikipedia.org/wiki/Quartile
void compute_summary(const std::vector<gko::size_type>& dist, json& out)
{
    const auto q = dist.size() / 4;
    const auto r = dist.size() % 4;
    // clang-format off
    const gko::size_type positions[4][6] = {
        { q - 1, q    , 2 * q - 1, 2 * q    , 3 * q - 1, 3 * q     },
        { q - 1, q    , 2 * q    , 2 * q    , 3 * q    , 3 * q + 1 },
        { q    , q    , 2 * q    , 2 * q + 1, 3 * q + 1, 3 * q + 1 },
        { q    , q + 1, 2 * q + 1, 2 * q + 1, 3 * q + 1, 3 * q + 2 }
    };
    const double coefs[4][6] = {
        { 0.5  , 0.5  , 0.5      , 0.5      , 0.5      , 0.5       },
        { 0.75 , 0.25 , 0.5      , 0.5      , 0.25     , 0.75      },
        { 0.5  , 0.5  , 0.5      , 0.5      , 0.5      , 0.5       },
        { 0.25 , 0.75 , 0.5      , 0.5      , 0.75     , 0.25      }
    };
    // clang-format on

    out["min"] = dist.front();
    out["q1"] = coefs[r][0] * static_cast<double>(dist[positions[r][0]]) +
                coefs[r][1] * static_cast<double>(dist[positions[r][1]]);
    out["median"] = coefs[r][2] * static_cast<double>(dist[positions[r][2]]) +
                    coefs[r][3] * static_cast<double>(dist[positions[r][3]]);
    out["q3"] = coefs[r][4] * static_cast<double>(dist[positions[r][4]]) +
                coefs[r][5] * static_cast<double>(dist[positions[r][5]]);
    out["max"] = dist.back();
}


double compute_moment(int degree, const std::vector<gko::size_type>& dist,
                      double center = 0.0, double normalization = 1.0)
{
    if (normalization == 0.0) {
        return 0.0;
    }
    double moment = 0.0;
    for (const auto& x : dist) {
        moment += std::pow(static_cast<double>(x) - center, degree);
    }
    return moment / static_cast<double>(dist.size()) /
           std::pow(normalization, static_cast<double>(degree));
}


// See en.wikipedia.org/wiki/Moment_(mathematics)
void compute_moments(const std::vector<gko::size_type>& dist, json& out)
{
    const auto mean = compute_moment(1, dist);
    out["mean"] = mean;
    const auto variance = compute_moment(2, dist, mean);
    out["variance"] = variance;
    const auto dev = std::sqrt(variance);
    out["skewness"] = compute_moment(3, dist, mean, dev);
    out["kurtosis"] = compute_moment(4, dist, mean, dev);
    out["hyperskewness"] = compute_moment(5, dist, mean, dev);
    out["hyperflatness"] = compute_moment(6, dist, mean, dev);
}


void compute_distribution_properties(const std::vector<gko::size_type>& dist,
                                     json& out)
{
    compute_summary(dist, out);
    compute_moments(dist, out);
}


void extract_matrix_statistics(gko::matrix_data<etype, gko::int64>& data,
                               json& problem)
{
    std::vector<gko::size_type> row_dist(data.size[0]);
    std::vector<gko::size_type> col_dist(data.size[1]);
    for (const auto& v : data.nonzeros) {
        ++row_dist[v.row];
        ++col_dist[v.column];
    }

    problem["rows"] = data.size[0];
    problem["columns"] = data.size[1];
    problem["nonzeros"] = data.nonzeros.size();

    std::sort(begin(row_dist), end(row_dist));
    problem["row_distribution"] = json::object();
    compute_distribution_properties(row_dist, problem["row_distribution"]);

    std::sort(begin(col_dist), end(col_dist));
    problem["col_distribution"] = json::object();
    compute_distribution_properties(col_dist, problem["col_distribution"]);
}


using Generator = DefaultSystemGenerator<etype, gko::int64>;


struct empty_state {};


struct MatrixStatistics : Benchmark<empty_state> {
    std::string name;
    std::vector<std::string> empty;

    MatrixStatistics() : name{"problem"} {}

    const std::string& get_name() const override { return name; }

    const std::vector<std::string>& get_operations() const override
    {
        return empty;
    }

    bool should_print() const override { return true; }

    std::string get_example_config() const override
    {
        return Generator::get_example_config();
    }

    bool validate_config(const json& test_case) const override
    {
        return Generator::validate_config(test_case);
    }

    std::string describe_config(const json& test_case) const override
    {
        return Generator::describe_config(test_case);
    }

    empty_state setup(std::shared_ptr<gko::Executor> exec,
                      json& test_case) const override
    {
        auto data = Generator::generate_matrix_data(test_case);
        // no reordering here, as it doesn't change statistics
        std::clog << "Matrix is of size (" << data.size[0] << ", "
                  << data.size[1] << "), " << data.nonzeros.size() << std::endl;
        test_case["rows"] = data.size[0];
        test_case["cols"] = data.size[1];
        test_case["nonzeros"] = data.nonzeros.size();

        extract_matrix_statistics(data, test_case["problem"]);
        return {};
    }


    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, empty_state& data,
             const std::string& operation_name,
             json& operation_case) const override
    {}
};


int main(int argc, char* argv[])
{
    std::string header =
        "A utility that collects additional statistical properties of the "
        "matrix.\n";
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format);

    std::clog << gko::version_info::get() << std::endl;

    auto test_cases = json::parse(get_input_stream());
    auto exec = gko::ReferenceExecutor::create();

    run_test_cases(MatrixStatistics{}, exec, get_timer(exec, false),
                   test_cases);

    std::cout << std::setw(4) << test_cases << std::endl;
}
