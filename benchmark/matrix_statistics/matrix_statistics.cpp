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

#include <ginkgo/ginkgo.hpp>


#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/spmv_common.hpp"


// some Ginkgo shortcuts
using etype = double;


// See en.wikipedia.org/wiki/Five-number_summary
// Quartile computation uses Method 3 from en.wikipedia.org/wiki/Quartile
void compute_summary(const std::vector<gko::size_type> &dist,
                     rapidjson::Value &out,
                     rapidjson::MemoryPoolAllocator<> &allocator)
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

    add_or_set_member(out, "min", dist[0], allocator);
    add_or_set_member(out, "q1",
                      coefs[r][0] * dist[positions[r][0]] +
                          coefs[r][1] * dist[positions[r][1]],
                      allocator);
    add_or_set_member(out, "median",
                      coefs[r][2] * dist[positions[r][2]] +
                          coefs[r][3] * dist[positions[r][3]],
                      allocator);
    add_or_set_member(out, "q3",
                      coefs[r][4] * dist[positions[r][4]] +
                          coefs[r][5] * dist[positions[r][5]],
                      allocator);
    add_or_set_member(out, "max", dist[dist.size() - 1], allocator);
}


double compute_moment(int degree, const std::vector<gko::size_type> &dist,
                      double center = 0.0, double normalization = 1.0)
{
    if (normalization == 0.0) {
        return 0.0;
    }
    auto moment = 0.0;
    for (const auto &x : dist) {
        moment += std::pow(x - center, degree);
    }
    return moment / dist.size() / std::pow(normalization, degree);
}


// See en.wikipedia.org/wiki/Moment_(mathematics)
void compute_moments(const std::vector<gko::size_type> &dist,
                     rapidjson::Value &out,
                     rapidjson::MemoryPoolAllocator<> &allocator)
{
    const auto mean = compute_moment(1, dist);
    add_or_set_member(out, "mean", mean, allocator);
    const auto variance = compute_moment(2, dist, mean);
    add_or_set_member(out, "variance", variance, allocator);
    const auto dev = std::sqrt(variance);
    add_or_set_member(out, "skewness", compute_moment(3, dist, mean, dev),
                      allocator);
    add_or_set_member(out, "kurtosis", compute_moment(4, dist, mean, dev),
                      allocator);
    add_or_set_member(out, "hyperskewness", compute_moment(5, dist, mean, dev),
                      allocator);
    add_or_set_member(out, "hyperflatness", compute_moment(6, dist, mean, dev),
                      allocator);
}


template <typename Allocator>
void compute_distribution_properties(const std::vector<gko::size_type> &dist,
                                     rapidjson::Value &out,
                                     Allocator &allocator)
{
    compute_summary(dist, out, allocator);
    compute_moments(dist, out, allocator);
}


template <typename Allocator>
void extract_matrix_statistics(gko::matrix_data<etype, gko::int64> &data,
                               rapidjson::Value &problem, Allocator &allocator)
{
    std::vector<gko::size_type> row_dist(data.size[0]);
    std::vector<gko::size_type> col_dist(data.size[1]);
    for (const auto &v : data.nonzeros) {
        ++row_dist[v.row];
        ++col_dist[v.column];
    }

    std::sort(begin(row_dist), end(row_dist));
    add_or_set_member(problem, "row_distribution",
                      rapidjson::Value(rapidjson::kObjectType), allocator);
    compute_distribution_properties(row_dist, problem["row_distribution"],
                                    allocator);

    std::sort(begin(col_dist), end(col_dist));
    add_or_set_member(problem, "col_distribution",
                      rapidjson::Value(rapidjson::kObjectType), allocator);
    compute_distribution_properties(col_dist, problem["col_distribution"],
                                    allocator);
}


int main(int argc, char *argv[])
{
    std::string header =
        "A utility that collects additional statistical properties of the "
        "matrix.\n";
    std::string format = std::string() + "  [\n" +
                         "    { \"filename\": \"my_file.mtx\"},\n" +
                         "    { \"filename\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::clog << gko::version_info::get() << std::endl;

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("problem")) {
                test_case.AddMember("problem",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &problem = test_case["problem"];

            std::clog << "Running test case: " << test_case << std::endl;

            std::ifstream ifs(test_case["filename"].GetString());
            auto matrix = gko::read_raw<etype, gko::int64>(ifs);
            ifs.close();

            std::clog << "Matrix is of size (" << matrix.size[0] << ", "
                      << matrix.size[1] << ")" << std::endl;

            extract_matrix_statistics(matrix, test_case["problem"], allocator);

            backup_results(test_cases);
        } catch (const std::exception &e) {
            std::cerr << "Error extracting statistics, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases;
}
