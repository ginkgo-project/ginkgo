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


const MachineTopology::io_obj_info *MachineTopology::get_pci_device(
    const std::string &pci_bus_id) const
{
    for (auto id = 0; id < this->pci_devices_.size(); ++id) {
        if (this->pci_devices_[id].pci_bus_id.compare(0, 12, pci_bus_id, 0,
                                                      12) == 0) {
            return &this->pci_devices_[id];
        }
    }
    return nullptr;
}


MachineTopology::MachineTopology()
{
#if GKO_HAVE_HWLOC

    // Initialize the topology from hwloc
    this->topo_ =
        hwloc_manager<hwloc_topology>(init_topology(), hwloc_topology_destroy);
    // load objects of type Package . See HWLOC_OBJ_PACKAGE for more details.
    load_objects(HWLOC_OBJ_PACKAGE, this->packages_);
    // load objects of type NUMA Node. See HWLOC_OBJ_NUMANODE for more details.
    load_objects(HWLOC_OBJ_NUMANODE, this->numa_nodes_);
    // load objects of type Core. See HWLOC_OBJ_CORE for more details.
    load_objects(HWLOC_OBJ_CORE, this->cores_);
    // load objects of type processing unit(PU). See HWLOC_OBJ_PU for more
    // details.
    load_objects(HWLOC_OBJ_PU, this->pus_);
    // load objects of type PCI Devices See HWLOC_OBJ_PCI_DEVICE for more
    // details.
    load_objects(HWLOC_OBJ_PCI_DEVICE, this->pci_devices_);
    num_numas_ = hwloc_get_nbobjs_by_type(this->topo_.get(), HWLOC_OBJ_PACKAGE);

#else

    this->topo_ = hwloc_manager<hwloc_topology>();

#endif
}


void MachineTopology::hwloc_binding_helper(
    const std::tuple<std::vector<MachineTopology::normal_obj_info>,
                     MachineTopology::hwloc_manager<hwloc_bitmap_s>> &obj,
    const int *id, const size_type num_ids,
    const MachineTopology::BitMapType &bitmap_type) const
{
#if GKO_HAVE_HWLOC
    auto bitmap_cur = hwloc_bitmap_alloc();
    auto bitmap_toset = hwloc_bitmap_alloc();
    auto bitmap_res = hwloc_bitmap_alloc();
    // Get current bitmap
    hwloc_get_cpubind(this->topo_.get(), bitmap_cur, 0);
    // Set the given ids to a new bitmap
    for (auto i = 0; i < num_ids; ++i) {
        GKO_ASSERT(id[i] < std::get<0>(obj).size());
        GKO_ASSERT(id[i] >= 0);
        hwloc_bitmap_set(bitmap_toset, std::get<0>(obj)[id[i]].os_id);
        // Check if this id has been bound before, Binding multiple threads to
        // same object may not be suitable.
        if (hwloc_bitmap_isset(std::get<1>(obj).get(),
                               std::get<0>(obj)[id[i]].os_id)) {
            hwloc_bitmap_clr(bitmap_toset, std::get<0>(obj)[id[i]].os_id);
        }
    }
    // Log the bound ids to the global bitmap.
    hwloc_bitmap_and(std::get<1>(obj).get(), std::get<1>(obj).get(),
                     bitmap_toset);
    // Depending on the passed in BitMapType, do the binary operation
    // Op(current_bitmap, toset_bitmap) and use the resulting bitmap to bind the
    // cpu.
    if (bitmap_type == BitMapType::bitmap_set) {
        hwloc_bitmap_free(bitmap_res);
        bitmap_res = hwloc_bitmap_dup(bitmap_toset);
    } else if (bitmap_type == BitMapType::bitmap_or) {
        hwloc_bitmap_or(bitmap_res, bitmap_cur, bitmap_toset);
    } else if (bitmap_type == BitMapType::bitmap_and) {
        hwloc_bitmap_and(bitmap_res, bitmap_cur, bitmap_toset);
    } else if (bitmap_type == BitMapType::bitmap_andnot) {
        hwloc_bitmap_andnot(bitmap_res, bitmap_cur, bitmap_toset);
    }

    // Singlify to reduce expensive migrations.
    hwloc_bitmap_singlify(bitmap_res);
    hwloc_set_cpubind(this->topo_.get(), bitmap_res, 0);
    hwloc_bitmap_free(bitmap_cur);
    hwloc_bitmap_free(bitmap_toset);
    hwloc_bitmap_free(bitmap_res);
#endif
}


void MachineTopology::hwloc_print_children(const hwloc_obj_t obj,
                                           const int depth)
{
#if GKO_HAVE_HWLOC
    char type[32], attr[1024];
    unsigned i;
    hwloc_obj_type_snprintf(type, sizeof(type), obj, 0);
    std::clog << std::string(2 * depth, ' ') << type;
    if (obj->os_index != (unsigned)-1) {
        std::clog << "#" << obj->os_index;
    }
    hwloc_obj_attr_snprintf(attr, sizeof(attr), obj, " ", 0);
    if (*attr) {
        std::clog << "(" << attr << ")";
    }
    std::clog << std::endl;
    for (i = 0; i < obj->arity; i++) {
        hwloc_print_children(obj->children[i], depth + 1);
    }
#endif
}


void MachineTopology::load_objects(
    hwloc_obj_type_t type,
    std::tuple<std::vector<MachineTopology::normal_obj_info>,
               MachineTopology::hwloc_manager<hwloc_bitmap_s>> &objects)
{
#if GKO_HAVE_HWLOC
    std::get<1>(objects) =
        hwloc_manager<hwloc_bitmap_s>(init_bitmap(), hwloc_bitmap_free);
    // Get the number of normal objects of a certain type (Core, PU, Machine
    // etc.).
    unsigned num_objects = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    for (unsigned i = 0; i < num_objects; i++) {
        // Get the actual normal object of the given type.
        hwloc_obj_t obj = hwloc_get_obj_by_type(this->topo_.get(), type, i);
        std::get<0>(objects).push_back(
            normal_obj_info{obj, obj->logical_index, obj->os_index,
                            obj->gp_index, hwloc_bitmap_first(obj->nodeset)});
    }
#endif
}


inline int MachineTopology::get_obj_local_id_by_os_index(
    std::vector<MachineTopology::normal_obj_info> &objects,
    size_type os_index) const
{
#if GKO_HAVE_HWLOC
    for (auto id = 0; id < objects.size(); ++id) {
        if (objects[id].os_id == os_index) {
            return id;
        }
    }
    return -1;
#endif
}


inline int MachineTopology::get_obj_local_id_by_gp_index(
    std::vector<MachineTopology::normal_obj_info> &objects,
    size_type gp_index) const
{
#if GKO_HAVE_HWLOC
    for (auto id = 0; id < objects.size(); ++id) {
        if (objects[id].gp_id == gp_index) {
            return id;
        }
    }
    return -1;
#endif
}


void MachineTopology::load_objects(
    hwloc_obj_type_t type, std::vector<MachineTopology::io_obj_info> &vector)
{
#if GKO_HAVE_HWLOC
    GKO_ASSERT(std::get<0>(this->cores_).size() != 0);
    GKO_ASSERT(std::get<0>(this->pus_).size() != 0);
    unsigned num_objects = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    for (unsigned i = 0; i < num_objects; i++) {
        // Get the actual PCI object.
        hwloc_obj_t obj = hwloc_get_obj_by_type(this->topo_.get(), type, i);
        // Get the non-IO ancestor (which is the closest and the one that can be
        // bound to) of the object.
        auto ancestor = hwloc_get_non_io_ancestor_obj(this->topo_.get(), obj);
        // Create the object.
        vector.push_back(
            io_obj_info{obj, obj->logical_index, obj->os_index, obj->gp_index,
                        hwloc_bitmap_first(ancestor->nodeset), ancestor});
        // Get the corresponding cpuset of the ancestor nodeset
        hwloc_cpuset_t ancestor_cpuset = hwloc_bitmap_alloc();
        hwloc_cpuset_from_nodeset(this->topo_.get(), ancestor_cpuset,
                                  ancestor->nodeset);
        // Find the cpu object closest to this device from the ancestor cpuset
        // and store its id for binding purposes
        vector.back().closest_cpu_id = get_obj_local_id_by_os_index(
            std::get<0>(this->pus_), hwloc_bitmap_first(ancestor_cpuset));
        // Get local id of the ancestor object.
        if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_PACKAGE) == 0) {
            vector.back().ancestor_local_id = get_obj_local_id_by_gp_index(
                std::get<0>(this->packages_), ancestor->gp_index);
        } else if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_CORE) == 0) {
            vector.back().ancestor_local_id = get_obj_local_id_by_gp_index(
                std::get<0>(this->cores_), ancestor->gp_index);
        } else if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_NUMANODE) ==
                   0) {
            vector.back().ancestor_local_id = get_obj_local_id_by_gp_index(
                std::get<0>(this->numa_nodes_), ancestor->gp_index);
        }
        hwloc_bitmap_free(ancestor_cpuset);
        // Get type of the ancestor object and store it as a string.
        char ances_type[24];
        hwloc_obj_type_snprintf(ances_type, sizeof(ances_type), ancestor, 0);
        vector.back().ancestor_type = std::string(ances_type);
        // Write the PCI Bus ID from the object info.
        char pci_bus_id[13];
        snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.%01x",
                 obj->attr->pcidev.domain, obj->attr->pcidev.bus,
                 obj->attr->pcidev.dev, obj->attr->pcidev.func);
        vector.back().pci_bus_id = std::string(pci_bus_id);
        // Get the number of IO children if any. For example, the software
        // OS devices (cuda, rsmi, ib) are listed as IO devices under a PCI
        // device.
        auto num_io_children = obj->io_arity;
        hwloc_obj_t os_child{};
        auto rem_io_children = num_io_children;
        // Get the IO children and their names and store them in vectors.
        if (num_io_children > 0) {
            // Get the first child
            os_child = obj->io_first_child;
            vector.back().io_children.push_back(os_child);
            vector.back().io_children_name.emplace_back(os_child->name);
            rem_io_children -= 1;
            // If the io obj has more than one children, they are siblings to
            // the first child. Store these as well.
            hwloc_obj_t os_child_n = os_child;
            while (rem_io_children >= 1) {
                os_child_n = os_child_n->next_sibling;
                vector.back().io_children.push_back(os_child_n);
                vector.back().io_children_name.emplace_back(os_child_n->name);
                rem_io_children -= 1;
            }
            GKO_ASSERT(vector.back().io_children.size() ==
                       vector.back().io_children_name.size());
        }
    }
#endif
}


hwloc_bitmap_s *MachineTopology::init_bitmap()
{
#if GKO_HAVE_HWLOC
    hwloc_bitmap_t tmp = hwloc_bitmap_alloc();
    return tmp;
#else
    // MSVC complains if there is no return statement.
    return nullptr;
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
