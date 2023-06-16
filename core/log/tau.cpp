/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#if GKO_HAVE_TAU
#define PERFSTUBS_USE_TIMERS
#include <perfstubs_api/timer.h>
#endif


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/log/profiler_hook.hpp>


namespace gko {
namespace log {


#if GKO_HAVE_TAU


void init_tau() { PERFSTUBS_INITIALIZE(); }


void begin_tau(const char* name, profile_event_category)
{
    PERFSTUBS_START_STRING(name);
}


void end_tau(const char*, profile_event_category)
{
    PERFSTUBS_STOP_STRING(name);
}


void finalize_tau() { PERFSTUBS_FINALIZE(); }


#else


void init_tau() GKO_NOT_COMPILED(tau);


void begin_tau(const char*, profile_event_category) GKO_NOT_COMPILED(tau);


void end_tau(const char*, profile_event_category) GKO_NOT_COMPILED(tau);


void finalize_tau() GKO_NOT_COMPILED(tau);


#endif


}  // namespace log
}  // namespace gko
