/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void residual_norm_kernel(
    size_type num_cols, ValueType rel_residual_goal,
    const ValueType *__restrict__ tau, const ValueType *__restrict__ orig_tau,
    uint8 stoppingId, bool setFinalized,
    stopping_status *__restrict__ stop_status,
    bool *__restrict__ device_storage)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_cols) {
        if (tau[tidx] < rel_residual_goal * orig_tau[tidx]) {
            stop_status[tidx].converge(stoppingId, setFinalized);
            device_storage[1] = true;
        }
        // because only false is written to all_converged, write conflicts
        // should not cause any problem
        else if (!stop_status[tidx].has_stopped()) {
            device_storage[0] = false;
        }
    }
}


__global__ __launch_bounds__(1) void init_kernel(
    bool *__restrict__ device_storage)
{
    device_storage[0] = true;
    device_storage[1] = false;
}
