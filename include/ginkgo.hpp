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

#ifndef GKO_GINKGO_HPP_
#define GKO_GINKGO_HPP_


#include "core/base/abstract_factory.hpp"
#include "core/base/array.hpp"
#include "core/base/exception.hpp"
#include "core/base/executor.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/math.hpp"
#include "core/base/matrix_data.hpp"
#include "core/base/mtx_reader.hpp"
#include "core/base/polymorphic_object.hpp"
#include "core/base/range.hpp"
#include "core/base/range_accessors.hpp"
#include "core/base/types.hpp"
#include "core/base/utils.hpp"

#include "core/log/record.hpp"
#include "core/log/stream.hpp"

#include "core/matrix/coo.hpp"
#include "core/matrix/csr.hpp"
#include "core/matrix/dense.hpp"
#include "core/matrix/ell.hpp"
#include "core/matrix/identity.hpp"

#include "core/preconditioner/block_jacobi.hpp"

#include "core/solver/bicgstab.hpp"
#include "core/solver/cg.hpp"
#include "core/solver/cgs.hpp"
#include "core/solver/fcg.hpp"

#include "core/stop/byinteraction.hpp"
#include "core/stop/combined.hpp"
#include "core/stop/iteration.hpp"
#include "core/stop/relative_residual_norm.hpp"
#include "core/stop/stopping_status.hpp"
#include "core/stop/time.hpp"


#endif  // GKO_GINKGO_HPP_
