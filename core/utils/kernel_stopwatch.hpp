#pragma once


#include <ginkgo/core/base/executor.hpp>


#include <chrono>
#include <map>
#include <memory>
#include <string>


namespace gko {


class kernel_stopwatch {
    using logger_type = std::map<std::string, std::chrono::duration<double>>;

public:
    kernel_stopwatch(const gko::Executor* exec, logger_type* logger)
        : exec_{exec}, logger_{logger}, start_{}
    {}
    void start()
    {
        if (logger_) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    void stop(const std::string& kernel_name)
    {
        if (logger_) {
            exec_->synchronize();
            const auto now = std::chrono::steady_clock::now();
            const std::chrono::duration<double> duration = now - start_;
            (*logger_)[kernel_name] += duration;
            start_ = std::chrono::steady_clock::now();
        }
    }


private:
    const gko::Executor* exec_;
    logger_type* logger_;
    std::chrono::steady_clock::time_point start_;
};


}  // namespace gko
