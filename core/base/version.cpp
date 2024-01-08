// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/version.hpp>


namespace gko {


version version_info::get_core_version() noexcept
{
    // When compiling the module, the header version is the same as the library
    // version. Mismatch between the header and the module versions may happen
    // if using shared libraries from different versions of Ginkgo.
    return version_info::get_header_version();
}


std::ostream& operator<<(std::ostream& os, const version_info& ver_info)
{
    auto print_version = [](std::ostream& os, const version& ver) -> void {
        static const std::string not_compiled_tag = "not compiled";
        if (ver.tag == not_compiled_tag) {
            os << "not compiled";
        } else {
            os << ver;
        }
    };

    os << "This is Ginkgo " << ver_info.header_version
       << "\n    running with core module " << ver_info.core_version
       << "\n    the reference module is  ";
    print_version(os, ver_info.reference_version);
    os << "\n    the OpenMP    module is  ";
    print_version(os, ver_info.omp_version);
    os << "\n    the CUDA      module is  ";
    print_version(os, ver_info.cuda_version);
    os << "\n    the HIP       module is  ";
    print_version(os, ver_info.hip_version);
    os << "\n    the DPCPP     module is  ";
    print_version(os, ver_info.dpcpp_version);
    return os;
}


}  // namespace gko
