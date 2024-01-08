// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_omp_version() noexcept
{
    // When compiling the module, the header version is the same as the library
    // version. Mismatch between the header and the module versions may happen
    // if using shared libraries from different versions of Ginkgo.
    return version_info::get_header_version();
}


}  // namespace gko
