// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>
#include <string>

#include "python_handler.hpp"


class CerebrasHandle {
public:
    CerebrasHandle(bool use_simulator = false);

    void copy_h2d(std::string target_var, float* vec, size_t vec_size,
                  int offset1, int offset2, int size1, int size2,
                  int elements_per_pe, bool streaming, bool nonblocking);

    void copy_d2h(std::string target_var, float* vec, size_t vec_size,
                  int offset1, int offset2, int size1, int size2,
                  int elements_per_pe, bool streaming, bool nonblocking);

    void call_func(std::string func_name, bool nonblocking = false);

    void destroy();

private:
    std::unique_ptr<PythonObject> pFuncDict_;

    // hacky numpy stuff - cannot be inlined as this is a macro that in
    // some cases expands to "return NULL;"
    void* init_numpy(void);

    // TODO: does this return a view on the array or a copy of the array?
    // I believe view, but not sure...
    PythonObject array_to_numpy(float* vec, size_t length);

    void start_device(bool use_simulator);

    void stop_device();
};


typedef struct CerebrasHandle* CerebrasContext;
