/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <iomanip>
#include <thread>


#include <ginkgo/core/base/executor.hpp>


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

std::string create_json(const std::string& resources)
{
    std::string json;
    json.append(R"({
  "version": {
    "major": 1,
    "minor": 0
  },
  "local": [
    {
)");
    for (const auto& line : split(resources, '\n')) {
        json.append(R"(      )");
        json.append(line);
        json.append("\n");
    }
    json.append(R"(    }
  ]
})");
    return json;
}


int main()
{
    auto num_cpu_threads = std::max(std::thread::hardware_concurrency(), 1u);
    auto num_cuda_gpus = gko::CudaExecutor::get_num_devices();
    auto num_hip_gpus = gko::HipExecutor::get_num_devices();
    auto num_sycl_gpus = gko::DpcppExecutor::get_num_devices("gpu");

    std::string cpus = R"("cpu": [{"id": "0", "slots": )" +
                       std::to_string(num_cpu_threads) + "}]";

    std::string gpus = "";
    auto add_devices = [&](int num_devices, const std::string& name) {
        if (num_devices) {
            gpus.append(",\n");
            gpus += '"' + name + "\": [\n";
        }
        for (int i = 0; i < num_devices; i++) {
            if (i > 0) {
                gpus.append(",\n");
            }
            gpus += R"(  {"id": ")" + std::to_string(i) + R"(", "slots": 100})";
        }
        if (num_devices) {
            gpus.append("\n]");
        }
    };
    add_devices(num_cuda_gpus, "cudagpu");
    add_devices(num_hip_gpus, "hipgpu");
    add_devices(num_sycl_gpus, "syclgpu");

    std::cout << create_json(cpus + gpus) << std::endl;
}
