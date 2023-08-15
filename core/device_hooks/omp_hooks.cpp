// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/scoped_device_id_guard.hpp>
#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_omp_version() noexcept
{
    // We just return the version with a special "not compiled" tag in
    // placeholder modules.
    return {GKO_VERSION_STR, "not compiled"};
}


scoped_device_id_guard::scoped_device_id_guard(const OmpExecutor* exec,
                                               int device_id)
    GKO_NOT_COMPILED(omp);


}  // namespace gko


#define GKO_HOOK_MODULE omp
#include "core/device_hooks/common_kernels.inc.cpp"
#undef GKO_HOOK_MODULE
