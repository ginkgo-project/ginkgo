#include "utils/executor.hpp"


#include "exception.hpp"
#include "exception_helpers.hpp"

int test() { auto cpu = msparse::CpuExecutor::create(); }
