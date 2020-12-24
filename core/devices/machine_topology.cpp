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
    // load objects of type processing unit(PU). See HWLOC_OBJ_TYPE for more
    // details.
    load_objects(HWLOC_OBJ_PU, this->pus_);
    // load objects of type PCI Devices See HWLOC_OBJ_TYPE for more
    // details.
    load_objects(HWLOC_OBJ_PCI_DEVICE, this->pci_devices_);
    num_numas_ = hwloc_get_nbobjs_by_type(this->topo_.get(), HWLOC_OBJ_PACKAGE);

#else

    this->topo_ = topo_manager<hwloc_topology>();

#endif
}


template <typename ObjInfoType>
void MachineTopology::hwloc_binding_helper(std::vector<ObjInfoType> &obj,
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


void MachineTopology::load_objects(
    hwloc_obj_type_t type,
    std::vector<MachineTopology::normal_obj_info> &vector)
{
#if GKO_HAVE_HWLOC
    // Get the number of normal objects of a certain type (Core, PU, Machine
    // etc.).
    unsigned nbcores = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    for (unsigned i = 0; i < nbcores; i++) {
        // Get the actual normal object of the given type.
        hwloc_obj_t obj = hwloc_get_obj_by_type(this->topo_.get(), type, i);
        vector.push_back(normal_obj_info{obj, hwloc_bitmap_first(obj->nodeset),
                                         obj->logical_index, obj->os_index,
                                         obj->gp_index});
    }
#endif
}


void MachineTopology::load_objects(
    hwloc_obj_type_t type, std::vector<MachineTopology::io_obj_info> &vector)
{
#if GKO_HAVE_HWLOC
    unsigned nbcores = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    for (unsigned i = 0; i < nbcores; i++) {
        // Get the actual PCI object.
        hwloc_obj_t obj = hwloc_get_obj_by_type(this->topo_.get(), type, i);
        // Get the non-IO ancestor (which is the closest and the one that can be
        // bound to) of the object.
        auto ancestor = hwloc_get_non_io_ancestor_obj(this->topo_.get(), obj);
        vector.push_back(io_obj_info{obj, obj->logical_index, obj->os_index,
                                     obj->gp_index, ancestor,
                                     hwloc_bitmap_first(ancestor->nodeset)});
        // Write the PCI Bus ID from the object info.
        char pci_busid[14];
        snprintf(pci_busid, sizeof(pci_busid), "%04x:%02x:%02x.%01x",
                 obj->attr->pcidev.domain, obj->attr->pcidev.bus,
                 obj->attr->pcidev.dev, obj->attr->pcidev.func);
        vector.back().pci_busid = std::string(pci_busid);
        // Get the number of IO children if any. For example, the software OS
        // devices (cuda, rsmi, ib) are listed as IO devices under a PCI device.
        auto num_io_children = obj->io_arity;
        hwloc_obj_t os_child{};
        auto rem_io_children = num_io_children;
        // Get the IO childrens and their names and store them in vectors.
        if (num_io_children > 0) {
            os_child = obj->io_first_child;
            vector.back().io_children.push_back(os_child);
            vector.back().io_children_name.push_back(os_child->name);
            rem_io_children -= 1;
            while (rem_io_children >= 1) {
                auto os_child_2 = os_child->next_cousin;
                vector.back().io_children.push_back(os_child_2);
                vector.back().io_children_name.push_back(os_child_2->name);
                rem_io_children -= 1;
            }
            GKO_ASSERT(vector.back().io_children.size() ==
                       vector.back().io_children_name.size());
        }
    }
#endif
}


hwloc_topology *MachineTopology::init_topology()
{
#if GKO_HAVE_HWLOC
    hwloc_topology_t tmp;
    hwloc_topology_init(&tmp);

    hwloc_topology_set_io_types_filter(tmp, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_set_type_filter(tmp, HWLOC_OBJ_BRIDGE,
                                   HWLOC_TYPE_FILTER_KEEP_NONE);
    hwloc_topology_set_type_filter(tmp, HWLOC_OBJ_OS_DEVICE,
                                   HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_set_xml(tmp, GKO_HWLOC_XMLFILE);
    hwloc_topology_load(tmp);

    return tmp;
#else
    // MSVC complains if there is no return statement.
    return nullptr;
#endif
}


}  // namespace gko
