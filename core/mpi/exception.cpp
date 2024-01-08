// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <string>


#include <mpi.h>


#include <ginkgo/core/base/exception.hpp>


namespace gko {


std::string MpiError::get_error(int64 error_code)
{
    int len{};
    std::array<char, MPI_MAX_ERROR_STRING> error_buf;
    MPI_Error_string(error_code, &error_buf[0], &len);
    std::string message = "MPI Error: " + std::string(&error_buf[0], len);

    return message;
}


}  // namespace gko
