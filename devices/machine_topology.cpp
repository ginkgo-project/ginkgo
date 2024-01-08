// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <atomic>
#include <memory>
#include <mutex>


#include <ginkgo/core/base/machine_topology.hpp>


namespace gko {


namespace detail {


class topo_bitmap {
public:
    using bitmap_type = hwloc_bitmap_s;
#if GKO_HAVE_HWLOC
    topo_bitmap() : bitmap(hwloc_bitmap_alloc()) {}
    ~topo_bitmap() { hwloc_bitmap_free(bitmap); }
#endif
    bitmap_type* get() { return bitmap; }

private:
    bitmap_type* bitmap;
};


hwloc_topology* init_topology()
{
#if GKO_HAVE_HWLOC
    hwloc_topology_t tmp;
    hwloc_topology_init(&tmp);

    hwloc_topology_set_io_types_filter(tmp, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_set_type_filter(tmp, HWLOC_OBJ_BRIDGE,
                                   HWLOC_TYPE_FILTER_KEEP_NONE);
    hwloc_topology_set_type_filter(tmp, HWLOC_OBJ_OS_DEVICE,
                                   HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_load(tmp);

    return tmp;
#else
    return nullptr;
#endif
}


}  // namespace detail


const machine_topology::io_obj_info* machine_topology::get_pci_device(
    const std::string& pci_bus_id) const
{
    for (size_type id = 0; id < this->pci_devices_.size(); ++id) {
        if (this->pci_devices_[id].pci_bus_id.compare(0, 12, pci_bus_id, 0,
                                                      12) == 0) {
            return &this->pci_devices_[id];
        }
    }
    return nullptr;
}


machine_topology::machine_topology()
{
#if GKO_HAVE_HWLOC

    // Initialize the topology from hwloc
    this->topo_ = hwloc_manager<hwloc_topology>(detail::init_topology(),
                                                hwloc_topology_destroy);
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


void machine_topology::hwloc_binding_helper(
    const std::vector<machine_topology::normal_obj_info>& obj,
    const std::vector<int>& bind_ids, const bool singlify) const
{
#if GKO_HAVE_HWLOC
    detail::topo_bitmap bitmap_toset;
    auto num_ids = bind_ids.size();
    auto id = bind_ids.data();
    // Set the given ids to a bitmap
    for (size_type i = 0; i < num_ids; ++i) {
        GKO_ASSERT(id[i] < obj.size());
        GKO_ASSERT(id[i] >= 0);
        hwloc_bitmap_set(bitmap_toset.get(), obj[id[i]].os_id);
    }

    // Singlify to reduce expensive migrations, if asked for.
    if (singlify) {
        hwloc_bitmap_singlify(bitmap_toset.get());
    }
    hwloc_set_cpubind(this->topo_.get(), bitmap_toset.get(), 0);
#endif
}


void machine_topology::load_objects(
    hwloc_obj_type_t type,
    std::vector<machine_topology::normal_obj_info>& objects) const
{
#if GKO_HAVE_HWLOC
    // Get the number of normal objects of a certain type (Core, PU, Machine
    // etc.).
    unsigned num_objects = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    objects.reserve(num_objects);
    for (unsigned i = 0; i < num_objects; i++) {
        // Get the actual normal object of the given type.
        hwloc_obj_t obj = hwloc_get_obj_by_type(this->topo_.get(), type, i);
        objects.push_back(normal_obj_info{obj, obj->logical_index,
                                          obj->os_index, obj->gp_index,
                                          hwloc_bitmap_first(obj->nodeset)});
    }
#endif
}


inline int machine_topology::get_obj_id_by_os_index(
    const std::vector<machine_topology::normal_obj_info>& objects,
    size_type os_index) const
{
#if GKO_HAVE_HWLOC
    for (size_type id = 0; id < objects.size(); ++id) {
        if (objects[id].os_id == os_index) {
            return id;
        }
    }
#endif
    return -1;
}


inline int machine_topology::get_obj_id_by_gp_index(
    const std::vector<machine_topology::normal_obj_info>& objects,
    size_type gp_index) const
{
#if GKO_HAVE_HWLOC
    for (size_type id = 0; id < objects.size(); ++id) {
        if (objects[id].gp_id == gp_index) {
            return id;
        }
    }
#endif
    return -1;
}


void machine_topology::load_objects(
    hwloc_obj_type_t type,
    std::vector<machine_topology::io_obj_info>& vector) const
{
#if GKO_HAVE_HWLOC
    GKO_ASSERT(this->cores_.size() != 0);
    GKO_ASSERT(this->pus_.size() != 0);
    unsigned num_objects = hwloc_get_nbobjs_by_type(this->topo_.get(), type);
    vector.reserve(num_objects);
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
        detail::topo_bitmap ancestor_cpuset;
        hwloc_cpuset_from_nodeset(this->topo_.get(), ancestor_cpuset.get(),
                                  ancestor->nodeset);
        // Find the cpu objects closest to this device from the ancestor cpuset
        // and store their ids for binding purposes
        int closest_pu_id = -1;
        int closest_os_id = hwloc_bitmap_first(ancestor_cpuset.get());
        // clang-format off
        hwloc_bitmap_foreach_begin(closest_os_id, ancestor_cpuset.get())
            closest_pu_id = get_obj_id_by_os_index(this->pus_, closest_os_id);
            vector.back().closest_pu_ids.push_back(closest_pu_id);
        hwloc_bitmap_foreach_end();
        // clang-format on

        // Get local id of the ancestor object.
        if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_PACKAGE) == 0) {
            vector.back().ancestor_local_id =
                get_obj_id_by_gp_index(this->packages_, ancestor->gp_index);
        } else if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_CORE) == 0) {
            vector.back().ancestor_local_id =
                get_obj_id_by_gp_index(this->cores_, ancestor->gp_index);
        } else if (hwloc_compare_types(ancestor->type, HWLOC_OBJ_NUMANODE) ==
                   0) {
            vector.back().ancestor_local_id =
                get_obj_id_by_gp_index(this->numa_nodes_, ancestor->gp_index);
        }
        // Get type of the ancestor object and store it as a string.
        char ances_type[24];
        hwloc_obj_type_snprintf(ances_type, sizeof(ances_type), ancestor, 0);
        vector.back().ancestor_type = std::string(ances_type);
        // Write the PCI Bus ID from the object info.
        char pci_bus_id[14];
        snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.%01x",
                 obj->attr->pcidev.domain, obj->attr->pcidev.bus,
                 obj->attr->pcidev.dev, obj->attr->pcidev.func);
        vector.back().pci_bus_id = std::string(pci_bus_id);
    }
#endif
}


}  // namespace gko
