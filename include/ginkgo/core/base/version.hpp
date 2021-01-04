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

#ifndef GKO_PUBLIC_CORE_BASE_VERSION_HPP_
#define GKO_PUBLIC_CORE_BASE_VERSION_HPP_


#include <ostream>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * This structure is used to represent versions of various Ginkgo modules.
 *
 * Version structures can be compared using the usual relational operators.
 *
 * @ingroup ginkgo_version
 */
struct version {
    constexpr version(const uint64 major, const uint64 minor,
                      const uint64 patch, const char *tag)
        : major{major}, minor{minor}, patch{patch}, tag{tag}
    {}

    /**
     * The major version number.
     */
    const uint64 major;

    /**
     * The minor version number.
     */
    const uint64 minor;

    /**
     * The patch version number.
     */
    const uint64 patch;

    /**
     * Addition tag string that describes the version in more detail.
     *
     * It does not participate in comparisons.
     */
    const char *const tag;
};

inline bool operator==(const version &first, const version &second)
{
    return first.major == second.major && first.minor == second.minor &&
           first.patch == second.patch;
}

inline bool operator!=(const version &first, const version &second)
{
    return !(first == second);
}

inline bool operator<(const version &first, const version &second)
{
    if (first.major < second.major) return true;
    if (first.major == second.major && first.minor < second.minor) return true;
    if (first.major == second.major && first.minor == second.minor &&
        first.patch < second.patch)
        return true;
    return false;
}

inline bool operator<=(const version &first, const version &second)
{
    return !(second < first);
}

inline bool operator>(const version &first, const version &second)
{
    return second < first;
}

inline bool operator>=(const version &first, const version &second)
{
    return !(first < second);
}

#undef GKO_ENABLE_VERSION_COMPARISON


/**
 * Prints version information to a stream.
 *
 * @param os  output stream
 * @param ver  version structure
 *
 * @return os
 */
inline std::ostream &operator<<(std::ostream &os, const version &ver)
{
    os << ver.major << "." << ver.minor << "." << ver.patch;
    if (ver.tag) {
        os << " (" << ver.tag << ")";
    }
    return os;
}


/**
 * Ginkgo uses version numbers to label new features and to communicate backward
 * compatibility guarantees:
 *
 * 1.  Versions with different major version number have incompatible
 *     interfaces (parts of the earlier interface may not be present anymore,
 *     and new interfaces can appear).
 * 2.  Versions with the same major number X, but different minor numbers Y1 and
 *     Y2 numbers keep the same interface as version X.0.0, but additions to
 *     the interface in X.0.0 present in X.Y1.0 may not be present in X.Y2.0
 *     and vice versa.
 * 3.  Versions with the same major an minor version numbers, but different
 *     patch numbers have exactly the same interface, but the functionality may
 *     be different (something that is not implemented or has a bug in an
 *     earlier version may have this implemented or fixed in a later version).
 *
 * This structure provides versions of different parts of Ginkgo: the headers,
 * the core and the kernel modules (reference, OpenMP, CUDA, HIP).
 * To obtain an instance of version_info filled with information about the
 * current version of Ginkgo, call the version_info::get() static method.
 */
class version_info {
public:
    /**
     * Returns an instance of version_info.
     *
     * @return an instance of version info
     */
    static const version_info &get()
    {
        static version_info info{};
        return info;
    }

    /**
     * Contains version information of the header files.
     */
    version header_version;

    /**
     * Contains version information of the core library.
     *
     * This is the version of the static/shared library called "ginkgo".
     */
    version core_version;

    /**
     * Contains version information of the reference module.
     *
     * This is the version of the static/shared library called
     * "ginkgo_reference".
     */
    version reference_version;

    /**
     * Contains version information of the OMP module.
     *
     * This is the version of the static/shared library called "ginkgo_omp".
     */
    version omp_version;

    /**
     * Contains version information of the CUDA module.
     *
     * This is the version of the static/shared library called "ginkgo_cuda".
     */
    version cuda_version;

    /**
     * Contains version information of the HIP module.
     *
     * This is the version of the static/shared library called "ginkgo_hip".
     */
    version hip_version;

    /**
     * Contains version information of the DPC++ module.
     *
     * This is the version of the static/shared library called "ginkgo_dpcpp".
     */
    version dpcpp_version;

private:
    static constexpr version get_header_version() noexcept
    {
        return version{GKO_VERSION_MAJOR, GKO_VERSION_MINOR, GKO_VERSION_PATCH,
                       GKO_VERSION_TAG};
    }

    static version get_core_version() noexcept;

    static version get_reference_version() noexcept;

    static version get_omp_version() noexcept;

    static version get_cuda_version() noexcept;

    static version get_hip_version() noexcept;

    static version get_dpcpp_version() noexcept;

    version_info()
        : header_version{get_header_version()},
          core_version{get_core_version()},
          reference_version{get_reference_version()},
          omp_version{get_omp_version()},
          cuda_version{get_cuda_version()},
          hip_version{get_hip_version()},
          dpcpp_version{get_dpcpp_version()}
    {}
};


/**
 * Prints library version information in human-readable format to a stream.
 *
 * @param os  output stream
 * @param ver_info  version information
 *
 * @return os
 */
std::ostream &operator<<(std::ostream &os, const version_info &ver_info);


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_VERSION_HPP_
