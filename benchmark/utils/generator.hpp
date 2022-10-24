/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GINKGO_BENCHMARK_UTILS_GENERATOR_HPP
#define GINKGO_BENCHMARK_UTILS_GENERATOR_HPP


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"


struct DefaultSystemGenerator {
    using Vec = vec<etype>;

    std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec,
        rapidjson::Value& config) const
    {
        std::ifstream mtx_fd(config["filename"].GetString());
        auto data = gko::read_generic_raw<etype, itype>(mtx_fd);
        return generate_matrix(std::move(exec), config, data);
    }

    std::shared_ptr<gko::LinOp> generate_matrix(
        std::shared_ptr<const gko::Executor> exec, rapidjson::Value& config,
        const gko::matrix_data<etype, itype>& data) const
    {
        return gko::share(::formats::matrix_factory(
            config["optimal"]["spmv"].GetString(), std::move(exec), data));
    }

    std::unique_ptr<Vec> generate_rhs(std::shared_ptr<const gko::Executor> exec,
                                      const gko::LinOp* system_matrix,
                                      rapidjson::Value& config) const
    {
        if (config.HasMember("rhs")) {
            std::ifstream rhs_fd{config["rhs"].GetString()};
            return gko::read<Vec>(rhs_fd, std::move(exec));
        } else {
            return ::generate_rhs(std::move(exec), system_matrix, engine);
        }
    }

    std::unique_ptr<Vec> generate_initial_guess(
        std::shared_ptr<const gko::Executor> exec,
        const gko::LinOp* system_matrix, const Vec* rhs) const
    {
        return ::generate_initial_guess(std::move(exec), system_matrix, rhs,
                                        engine);
    }

    std::unique_ptr<Vec> initialize(
        std::initializer_list<etype> il,
        std::shared_ptr<const gko::Executor> exec) const
    {
        return gko::initialize<Vec>(std::move(il), std::move(exec));
    }

    std::default_random_engine engine = get_engine();
};

#endif  // GINKGO_BENCHMARK_UTILS_GENERATOR_HPP
