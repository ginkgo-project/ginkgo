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

#ifndef GKO_CORE_BASE_MACHINE_TOPOLOGY_HPP_
#define GKO_CORE_BASE_MACHINE_TOPOLOGY_HPP_


#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#if GKO_HAVE_HWLOC

#include <hwloc.h>

#else

struct hwloc_obj_type_t {};
struct hwloc_obj_t {};

#endif


struct hwloc_topology;


namespace gko {


class MachineTopology;

extern const MachineTopology *get_machine_topology();

extern std::shared_ptr<const MachineTopology> get_shared_machine_topology();


/**
 * The machine topology class represents the heirarchical topology of a machine,
 * including NUMA nodes, cores and GPUs. Various infomation of the machine are
 * gathered with the help of the Hardware Locality library (hwloc).
 *
 * This class also provides functionalities to bind objects in the topology to
 * the execution objects. Binding can enhance performance by allowing data to be
 * closer to the executing object.
 */
class MachineTopology {
    friend const MachineTopology *get_machine_topology();

private:
    /**
     * This struct holds the attributes for an object.
     */
    struct mach_topo_obj_info {
        /**
         * The hwloc object.
         */
        hwloc_obj_t obj;

        /**
         * The numa number of the object.
         */
        int numa;

        /**
         * The logical_id assigned by the OS.
         */
        size_type logical_id;

        /**
         * The physical_id assigned to the object.
         * For GPUs, this is their number in the numa
         */
        size_type physical_id;

        /**
         * The physical_id assigned to the object.
         * For GPUs, this is their number in the numa
         */
        size_type memory_size;
    };

public:
    /**
     * Creates a new MachineTopology object.
     */
    static std::shared_ptr<MachineTopology> create()
    {
        return std::shared_ptr<MachineTopology>(new MachineTopology());
    }

    MachineTopology(MachineTopology &) = delete;
    MachineTopology(MachineTopology &&) = delete;

    /**
     * Bind the object associated with the id to a core.
     *
     * @param id  The id of the object to be bound.
     */
    void bind_to_core(size_type id) { hwloc_binding_helper(this->cores_, id); }

    /**
     * Bind the object associated with the id to a Processing unit(PU).
     *
     * @param id  The id of the object to be bound.
     */
    void bind_to_pu(size_type id) { hwloc_binding_helper(this->pus_, id); }

    /**
     * Get the object of type PU associated with the id.
     *
     * @param id  The id of the PU
     */
    const mach_topo_obj_info &get_pu(size_type id)
    {
        GKO_ENSURE_IN_BOUNDS(id, pus_.size());
        return pus_[id];
    }


    /**
     * Get the object of type core associated with the id.
     *
     * @param id  The id of the core
     */
    const mach_topo_obj_info &get_core(size_type id)
    {
        GKO_ENSURE_IN_BOUNDS(id, cores_.size());
        return cores_[id];
    }


    /**
     * Get the object of type gpu associated with the id.
     *
     * @param id  The id of the gpu
     */
    const mach_topo_obj_info &get_gpu(size_type id)
    {
        GKO_ENSURE_IN_BOUNDS(id, gpus_.size());
        return gpus_[id];
    }


    /**
     * Get the number of PU objects stored in this Topology tree.
     */
    size_type get_num_pus() { return pus_.size(); }


    /**
     * Get the number of core objects stored in this Topology tree.
     */
    size_type get_num_cores() { return cores_.size(); }


    /**
     * Get the number of GPU objects stored in this Topology tree.
     */
    size_type get_num_gpus() { return gpus_.size(); }


    /**
     * Get the number of NUMA objects stored in this Topology tree.
     */
    size_type get_num_numas() { return num_numas_; }


    /**
     * Load the gpu objects. These functions are implemened by the respective
     * GPU Executor classes.
     */
    virtual void load_gpus() {}

protected:
    MachineTopology()
    {
#if GKO_HAVE_HWLOC


        this->topo_ = topo_manager<hwloc_topology>(init_topology(),
                                                   hwloc_topology_destroy);

        load_objects(HWLOC_OBJ_CORE, this->cores_);
        load_objects(HWLOC_OBJ_PU, this->pus_);
        this->load_gpus();

        num_numas_ = hwloc_get_nbobjs_by_type(topo_.get(), HWLOC_OBJ_PACKAGE);
#else

        this->topo_ = topo_manager<hwloc_topology>();

#endif
    }


    /**
     * A helper function that binds the object with an id.
     */
    void hwloc_binding_helper(std::vector<mach_topo_obj_info> &obj,
                              size_type id)
    {
#if GKO_HAVE_HWLOC
        auto bitmap = hwloc_bitmap_alloc();
        hwloc_bitmap_set(bitmap, obj[id].physical_id);
        hwloc_bitmap_singlify(bitmap);
        hwloc_set_cpubind(topo_.get(), bitmap, 0);
        hwloc_bitmap_free(bitmap);
#endif
    }


#if GKO_HAVE_HWLOC


    /**
     * A helper function that prints the topology tree of an object to a given
     * depth. Provided from the hwloc library.
     */
    static void hwloc_print_children(hwloc_topology *topology, hwloc_obj_t obj,
                                     int depth)
    {
        char type[32], attr[1024];
        unsigned i;
        hwloc_obj_type_snprintf(type, sizeof(type), obj, 0);
        std::cout << std::string(2 * depth, ' ') << type;
        if (obj->os_index != (unsigned)-1) {
            std::cout << "#" << obj->os_index;
        }
        hwloc_obj_attr_snprintf(attr, sizeof(attr), obj, " ", 0);
        if (*attr) {
            std::cout << "(" << attr << ")";
        }
        std::cout << std::endl;
        for (i = 0; i < obj->arity; i++) {
            hwloc_print_children(topology, obj->children[i], depth + 1);
        }
    }


    // The objects should be sorted by logical index since hwloc uses logical
    // index with these functions
    void load_objects(hwloc_obj_type_t type,
                      std::vector<mach_topo_obj_info> &vector)
    {
        unsigned nbcores = hwloc_get_nbobjs_by_type(topo_.get(), type);
        for (unsigned i = 0; i < nbcores; i++) {
            hwloc_obj_t obj = hwloc_get_obj_by_type(topo_.get(), type, i);
            vector.push_back(
                mach_topo_obj_info{obj, hwloc_bitmap_first(obj->nodeset),
                                   obj->logical_index, obj->os_index});
        }
    }


    hwloc_topology *init_topology()
    {
        hwloc_topology_t tmp;
        hwloc_topology_init(&tmp);

        hwloc_topology_set_io_types_filter(tmp,
                                           HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
        hwloc_topology_set_xml(tmp, GKO_HWLOC_XMLFILE);
        hwloc_topology_load(tmp);

        return tmp;
    }


#endif


private:
    std::vector<mach_topo_obj_info> gpus_;
    std::vector<mach_topo_obj_info> pus_;
    std::vector<mach_topo_obj_info> cores_;
    std::vector<mach_topo_obj_info> pci_devices_;
    size_type num_numas_;

    template <typename T>
    using topo_manager = std::unique_ptr<T, std::function<void(T *)>>;
    topo_manager<hwloc_topology> topo_;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_MACHINE_TOPOLOGY_HPP_
