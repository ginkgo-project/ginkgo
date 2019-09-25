/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_MACHINE_CONFIG_HPP_
#define GKO_CORE_BASE_MACHINE_CONFIG_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>

#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

// #include<cuda_runtime.h>

#if GKO_HAVE_HWLOC
#include <hwloc.h>
// #include <hwloc/cuda.h>
#endif


struct hwloc_topology;

namespace gko {
namespace machine_config {

struct hwloc_obj_info {
    int numa;
    std::size_t logical_id;
    std::size_t physical_id;  // for GPUs, this is their number in the numa
};


struct machine_information {
    virtual const hwloc_obj_info &get_pu(std::size_t id) = 0;
    virtual const hwloc_obj_info &get_core(std::size_t id) = 0;
    virtual const hwloc_obj_info &get_gpu(std::size_t id) = 0;
    virtual std::size_t get_num_pu() = 0;
    virtual std::size_t get_num_core() = 0;
    virtual std::size_t get_num_gpu() = 0;
    virtual std::size_t get_num_numa() = 0;
};


struct binder {
    virtual void bind_to_core(std::size_t id) = 0;
    virtual void bind_to_pu(std::size_t id) = 0;
};


class topology : public machine_information, public binder {
public:
    topology(topology &) = delete;
    topology(topology &&) = delete;

    static std::unique_ptr<topology> create()
    {
        return std::unique_ptr<topology>(new topology());
    }


    void bind_to_core(std::size_t id) override
    {
        hwloc_binding_helper(this->cores_, id);
    }

    void bind_to_pu(std::size_t id) override
    {
        hwloc_binding_helper(this->pus_, id);
    }

    const hwloc_obj_info &get_pu(std::size_t id) override
    {
        GKO_ENSURE_IN_BOUNDS(id, pus_.size());
        return pus_[id];
    }

    const hwloc_obj_info &get_core(std::size_t id) override
    {
        GKO_ENSURE_IN_BOUNDS(id, cores_.size());
        return cores_[id];
    }

    const hwloc_obj_info &get_gpu(std::size_t id) override
    {
        GKO_ENSURE_IN_BOUNDS(id, gpus_.size());
        return gpus_[id];
    }


    std::size_t get_num_pu() override { return pus_.size(); };
    std::size_t get_num_core() override { return cores_.size(); };
    std::size_t get_num_gpu() override { return gpus_.size(); };
    std::size_t get_num_numa() override { return num_numas_; };


protected:
    void hwloc_binding_helper(std::vector<hwloc_obj_info> &obj, std::size_t id)
    {
        // auto bitmap = hwloc_bitmap_alloc();
        // hwloc_bitmap_set(bitmap, obj[id].physical_id);
        // hwloc_set_cpubind(topo_.get(), bitmap, 0);
        // hwloc_bitmap_free(bitmap);
    }

    topology()
    {
        // this->topo_ = topo_manager<hwloc_topology>(init_topology(),
        //                                            hwloc_topology_destroy);

        // load_objects(HWLOC_OBJ_CORE, this->cores_);
        // load_objects(HWLOC_OBJ_PU, this->pus_);
        // load_gpus();

        // num_numas_ = hwloc_get_nbobjs_by_type(topo_.get(),
        // HWLOC_OBJ_PACKAGE);
    }


private:
    // The objects should be sorted by logical index since hwloc uses logical
    // index with these functions
    void load_objects(hwloc_obj_type_t type,
                      std::vector<hwloc_obj_info> &vector)
    {
        // unsigned nbcores = hwloc_get_nbobjs_by_type(topo_.get(), type);
        // for (unsigned i = 0; i < nbcores; i++) {
        //     hwloc_obj_t obj = hwloc_get_obj_by_type(topo_.get(), type, i);
        //     vector.push_back(hwloc_obj_info{hwloc_bitmap_first(obj->nodeset),
        //                                     obj->logical_index,
        //                                     obj->os_index});
        // }
    }

    void load_gpus()
    {
        // int ngpus = 0;
        // if (cudaGetDeviceCount(&ngpus) == cudaSuccess) {
        //     std::size_t num_in_numa = 0;
        //     int last_numa = 0;
        //     for (std::size_t i = 0; i < ngpus; i++, num_in_numa++) {
        //         auto obj = hwloc_cuda_get_device_osdev(topo_.get(), i);
        //         while (obj &&
        //                (!obj->nodeset || hwloc_bitmap_iszero(obj->nodeset)))
        //             obj = obj->parent;
        //         if (obj && obj->nodeset) {
        //             auto this_numa = hwloc_bitmap_first(obj->nodeset);
        //             if (this_numa != last_numa) {
        //                 num_in_numa = 0;
        //             }
        //             gpus_.push_back(hwloc_obj_info{this_numa, i,
        //             num_in_numa}); last_numa = this_numa;
        //         }
        //     }
        // }
    }

    hwloc_topology *init_topology()
    {
        // hwloc_topology_t tmp;
        // hwloc_topology_init(&tmp);
        // hwloc_topology_set_io_types_filter(tmp,
        //                                    HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
        // hwloc_topology_load(tmp);
        // return tmp;
    }

    std::vector<hwloc_obj_info> gpus_;
    std::vector<hwloc_obj_info> pus_;
    std::vector<hwloc_obj_info> cores_;
    std::size_t num_numas_;

    template <typename T>
    using topo_manager = std::unique_ptr<T, std::function<void(T *)>>;
    topo_manager<hwloc_topology> topo_;
};
}  // namespace machine_config
}  // namespace gko


#endif  // GKO_CORE_BASE_MACHINE_CONFIG_HPP_
