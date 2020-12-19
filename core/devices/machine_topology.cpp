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

#include <atomic>
#include <memory>
#include <mutex>


#include <ginkgo/core/base/machine_topology.hpp>


namespace gko {


namespace detail {


std::shared_ptr<const MachineTopology> machine_topology{};
std::mutex machine_topology_mutex{};
std::atomic<bool> initialized_machine_topology{};


}  // namespace detail


const MachineTopology *get_machine_topology()
{
    if (!detail::initialized_machine_topology.load()) {
        std::lock_guard<std::mutex> guard(detail::machine_topology_mutex);
        if (!detail::machine_topology) {
            detail::machine_topology = MachineTopology::create();
            detail::initialized_machine_topology.store(true);
        }
    }
    assert(detail::machine_topology.get() != nullptr);
    return detail::machine_topology.get();
}


MachineTopology::MachineTopology()
{
#if GKO_HAVE_HWLOC

    // Initialize the topology from hwloc
    this->topo_ =
        topo_manager<hwloc_topology>(init_topology(), hwloc_topology_destroy);

    // load objects of type CORE. See HWLOC_OBJ_TYPE for more details.
    load_objects(HWLOC_OBJ_CORE, this->cores_);
    // load objects of type PU. See HWLOC_OBJ_TYPE for more details.
    load_objects(HWLOC_OBJ_PU, this->pus_);
    // load_objects(HWLOC_OBJ_PCI_DEVICE, this->pci_devices_);

    num_numas_ = hwloc_get_nbobjs_by_type(this->topo_.get(), HWLOC_OBJ_PACKAGE);
#else

    this->topo_ = topo_manager<hwloc_topology>();

#endif
}


void MachineTopology::hwloc_binding_helper(std::vector<mach_topo_obj_info> &obj,
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


void MachineTopology::hwloc_print_children(hwloc_topology *topology,
                                           hwloc_obj_t obj, int depth)
{
#if GKO_HAVE_HWLOC
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
#endif
}


void MachineTopology::load_objects(hwloc_obj_type_t type,
                                   std::vector<mach_topo_obj_info> &vector)
{
#if GKO_HAVE_HWLOC
    unsigned nbcores = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    for (unsigned i = 0; i < nbcores; i++) {
        hwloc_obj_t obj = hwloc_get_obj_by_type(this->topo_.get(), type, i);
        vector.push_back(mach_topo_obj_info{obj,
                                            hwloc_bitmap_first(obj->nodeset),
                                            obj->logical_index, obj->os_index});
    }
#endif
}


hwloc_topology *MachineTopology::init_topology()
{
#if GKO_HAVE_HWLOC
    hwloc_topology_t tmp;
    hwloc_topology_init(&tmp);

    hwloc_topology_set_io_types_filter(tmp, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_set_xml(tmp, GKO_HWLOC_XMLFILE);
    hwloc_topology_load(tmp);

    return tmp;
#else
    // MSVC complains if there is no return statement.
    return nullptr;
#endif
}


}  // namespace gko
