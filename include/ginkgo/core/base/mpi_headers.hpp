/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_BASE_MPI_HEADERS_HPP_
#define GKO_PUBLIC_CORE_BASE_MPI_HEADERS_HPP_


#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


#ifndef MPI_VERSION

using MPI_Comm = int;
using MPI_Status = int;
using MPI_Request = int;
using MPI_Datatype = int;
using MPI_Op = int;
using MPI_Win = int *;
using MPI_Info = int *;

#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif
#ifndef MPI_COMM_SELF
#define MPI_COMM_SELF 0
#endif
#ifndef MPI_REQUEST_NULL
#define MPI_REQUEST_NULL 0
#endif
#ifndef MPI_INFO_NULL
#define MPI_INFO_NULL 0
#endif
#ifndef MPI_MIN
#define MPI_MIN 0
#endif
#ifndef MPI_MAX
#define MPI_MAX 0
#endif
#ifndef MPI_SUM
#define MPI_SUM 0
#endif
#endif


namespace gko {
namespace mpi {

enum class op_type {
    sum = 1,
    min = 2,
    max = 3,
    product = 4,
    custom = 5,
    logical_and = 6,
    bitwise_and = 7,
    logical_or = 8,
    bitwise_or = 9,
    logical_xor = 10,
    bitwise_xor = 11,
    max_val_and_loc = 12,
    min_val_and_loc = 13
};


/*
 * Class that allows an RAII of initialization and calls MPI_Finalize at the
 * end of its scope. Therefore, this must be called before an MpiExecutor is
 * created.
 */
class init_finalize {
public:
    init_finalize(int &argc, char **&argv, const size_type num_threads);

    init_finalize() = delete;

    init_finalize(init_finalize &other) = delete;

    init_finalize &operator=(const init_finalize &other) = delete;

    init_finalize(init_finalize &&other) = delete;

    init_finalize const &operator=(init_finalize &&other) = delete;

    static bool is_finalized();

    static bool is_initialized();

    ~init_finalize() noexcept(false);

private:
    int num_args_;
    int required_thread_support_;
    int provided_thread_support_;
    char **args_;
};


/**
 * A class holding and operating on the MPI_Info class. Stores the key value
 * pair as a map and provides methods to access these values with keys as
 * strings.
 */
class info {
    info();

    void remove(std::string key);

    std::string &at(std::string &key) { return this->key_value_.at(key); }

    void add(std::string key, std::string value);

    MPI_Info get() { return this->info_; }

    ~info();

private:
    std::map<std::string, std::string> key_value_;
    MPI_Info info_;
};


/**
 * A communicator class that takes in the given communicator and duplicates it
 * for our purposes. As the class or object goes out of scope, the communicator
 * is freed.
 */
class communicator {
public:
    communicator(const MPI_Comm &comm);

    communicator(const MPI_Comm &comm, int color, int key);

    communicator() = delete;

    communicator(communicator &other) = delete;

    communicator &operator=(const communicator &other) = delete;

    communicator(communicator &&other) = default;

    communicator &operator=(communicator &&other) = default;

    MPI_Comm get() const { return comm_; }

    bool compare(const MPI_Comm &other) const;

    ~communicator();

private:
    MPI_Comm comm_;
};


template <typename ValueType>
class window {
public:
    enum class win_type { allocate = 1, create = 2, dynamic_create = 3 };
    enum class lock_type { shared = 1, exclusive = 2 };

    window() : window_(nullptr) {}
    window(window &other) = delete;
    window &operator=(const window &other) = delete;
    window(window &&other) = default;
    window &operator=(window &&other) = default;

    window(ValueType *base, unsigned int size,
           const int disp_unit = sizeof(ValueType),
           MPI_Info info = MPI_INFO_NULL, const MPI_Comm &comm = MPI_COMM_WORLD,
           win_type create_type = win_type::create);

    MPI_Win get() { return this->window_; }

    void fence(int assert = 0);

    void lock(int rank, int assert = 0, lock_type lock_t = lock_type::shared);

    void unlock(int rank);

    void lock_all(int assert = 0);

    void unlock_all();

    void flush(int rank);

    void flush_local(int rank);

    void flush_all();

    void flush_all_local();

    ~window();

private:
    MPI_Win window_;
};


}  // namespace mpi
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_MPI_HEADERS_HPP_
