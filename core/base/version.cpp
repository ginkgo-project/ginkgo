/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/base/version.hpp"


namespace gko {


version version_info::get_core_version() noexcept
{
    // When compiling the module, the header version is the same as the library
    // version. Mismatch between the header and the module versions may happen
    // if using shared libraries from different versions of Ginkgo.
    return version_info::get_header_version();
}


std::ostream &operator<<(std::ostream &os, const version_info &ver_info)
{
    auto print_version = [](std::ostream &os, const version &ver) -> void {
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
    os << "\n    the GPU       module is  ";
    print_version(os, ver_info.cuda_version);
    return os;
}


}  // namespace gko
