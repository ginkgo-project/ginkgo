#include <ginkgo/core/base/executor.hpp>

#include <iomanip>
#include <thread>


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
        if(num_devices){
            gpus.append(",\n");
            gpus += '"' + name + "\": [\n";
        }
        for (int i = 0; i < num_devices; i++) {
            if(i > 0){
                gpus.append(",\n");
            }
            gpus+= R"(  {"id": ")" + std::to_string(i) + R"(", "slots": 100})";
        }
        if(num_devices){
            gpus.append("\n]");
        }
    };
    add_devices(num_cuda_gpus, "cudagpu");
    add_devices(num_hip_gpus, "hipgpu");
    add_devices(num_sycl_gpus, "syclgpu");

    std::cout << create_json(cpus + gpus) << std::endl;
}
