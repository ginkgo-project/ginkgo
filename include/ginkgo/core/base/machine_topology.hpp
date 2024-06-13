// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_MACHINE_TOPOLOGY_HPP_
#define GKO_PUBLIC_CORE_BASE_MACHINE_TOPOLOGY_HPP_


#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
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
struct hwloc_bitmap_s;


namespace gko {


/**
 * The machine topology class represents the hierarchical topology of a machine,
 * including NUMA nodes, cores and PCI Devices. Various information of the
 * machine are gathered with the help of the Hardware Locality library (hwloc).
 *
 * This class also provides functionalities to bind objects in the topology to
 * the execution objects. Binding can enhance performance by allowing data to be
 * closer to the executing object.
 *
 * See the hwloc documentation
 * (https://www.open-mpi.org/projects/hwloc/doc/) for more detailed
 * information on topology detection and binding interfaces.
 *
 * @note A global object of machine_topology type is created in a thread safe
 *       manner and only destroyed at the end of the program. This means that
 *       any subsequent queries will be from the same global object and hence
 *       use an extra atomic read.
 */
class machine_topology {
    template <typename T>
    using hwloc_manager = std::unique_ptr<T, std::function<void(T*)>>;

    /**
     * This struct holds the attributes for a normal non-IO object.
     */
    struct normal_obj_info {
        /**
         * The hwloc object.
         */
        hwloc_obj_t obj;

        /**
         * The logical_id assigned by HWLOC (assigned according to physical
         * proximity).
         *
         * @note Use this rather than os_id for all purposes other than binding.
         *       [Reference](https://www.open-mpi.org/projects/hwloc/doc/v2.4.0/a00364.php#faq_indexes)
         */
        size_type logical_id;

        /**
         * The os_id assigned by the OS (assigned arbitrarily by the OS)
         */
        size_type os_id;

        /**
         * The global persistent id assigned to the object by hwloc.
         */
        size_type gp_id;

        /**
         * The numa number of the object.
         */
        int numa;

        /**
         * The memory size of the object.
         */
        size_type memory_size;
    };


    /**
     * This struct holds the attributes for an IO/Misc object.
     *
     * Mainly used for PCI devices. The identifier important for PCI devices is
     * the PCI Bus ID, stored here as a string. PCI devices themselves usually
     * contain Hard disks, network components as well as other objects that are
     * not important for our case.
     *
     * In many cases, hwloc is able to identify the OS devices that belong to a
     * certain PCI Bus ID and here they are stored in the io children vector. A
     * list of their names are also additionally stored for easy access and
     * comparison.
     *
     * @note IO children can have names such as ibX for Infiniband cards, cudaX
     * for NVIDIA cards with CUDA and rsmiX for AMD cards.
     */
    struct io_obj_info {
        /**
         * The hwloc object.
         */
        hwloc_obj_t obj;

        /**
         * The logical_id assigned by HWLOC (assigned according to proximity).
         *
         * @note Use this rather than os_id for all purposes other than binding.
         * [Reference](https://www.open-mpi.org/projects/hwloc/doc/v2.4.0/a00364.php#faq_indexes)
         */
        size_type logical_id;

        /**
         * The os_id assigned by the OS (assigned arbitrarily by the OS)
         */
        size_type os_id;

        /**
         * The global persistent id assigned to the object by hwloc.
         */
        size_type gp_id;

        /**
         * The closest numa.
         */
        int closest_numa;

        /**
         * The non-io parent object.
         */
        hwloc_obj_t non_io_ancestor;

        /**
         * The ancestor local id.
         */
        int ancestor_local_id;

        /**
         * The ancestor type.
         */
        std::string ancestor_type;

        /**
         * The array of CPU ids closest to the object.
         */
        std::vector<int> closest_pu_ids;

        /**
         * The PCI Bus ID
         */
        std::string pci_bus_id;
    };

public:
    /**
     * Returns an instance of the machine_topology object.
     *
     * @return  the machine_topology instance
     */
    static machine_topology* get_instance()
    {
        static machine_topology instance;
        return &instance;
    }

    /**
     * Bind the calling process to the CPU cores associated with
     * the ids.
     *
     * @param ids  The ids of cores to be bound.
     * @param singlify  The ids of PUs are singlified to prevent possibly
     *                  expensive migrations by the OS. This means that the
     *                  binding is performed for only one of the ids in the
     *                  set of ids passed in.
     *                  See hwloc doc for
     *                  [singlify](https://www.open-mpi.org/projects/hwloc/doc/v2.4.0/a00175.php#gaa611a77c092e679246afdf9a60d5db8b)
     */
    void bind_to_cores(const std::vector<int>& ids,
                       const bool singlify = true) const
    {
        hwloc_binding_helper(this->cores_, ids, singlify);
    }

    /**
     * Bind to a single core
     *
     * @param ids  The ids of the core to be bound to the calling process.
     */
    void bind_to_core(const int& id) const
    {
        machine_topology::get_instance()->bind_to_cores(std::vector<int>{id});
    }

    /**
     * Bind the calling process to PUs associated with
     * the ids.
     *
     * @param ids  The ids of PUs to be bound.
     * @param singlify  The ids of PUs are singlified to prevent possibly
     *                  expensive migrations by the OS. This means that the
     *                  binding is performed for only one of the ids in the
     *                  set of ids passed in.
     *                  See hwloc doc for
     *                  [singlify](https://www.open-mpi.org/projects/hwloc/doc/v2.4.0/a00175.php#gaa611a77c092e679246afdf9a60d5db8b)
     */
    void bind_to_pus(const std::vector<int>& ids,
                     const bool singlify = true) const
    {
        hwloc_binding_helper(this->pus_, ids, singlify);
    }

    /**
     * Bind to a Processing unit (PU)
     *
     * @param ids  The ids of PUs to be bound to the calling process.
     */
    void bind_to_pu(const int& id) const
    {
        machine_topology::get_instance()->bind_to_pus(std::vector<int>{id});
    }

    /**
     * Get the object of type PU associated with the id.
     *
     * @param id  The id of the PU
     * @return  the PU object struct.
     */
    const normal_obj_info* get_pu(size_type id) const
    {
        GKO_ENSURE_IN_BOUNDS(id, this->pus_.size());
        return &this->pus_[id];
    }

    /**
     * Get the object of type core associated with the id.
     *
     * @param id  The id of the core
     * @return  the core object struct.
     */
    const normal_obj_info* get_core(size_type id) const
    {
        GKO_ENSURE_IN_BOUNDS(id, this->cores_.size());
        return &this->cores_[id];
    }

    /**
     * Get the object of type pci device associated with the id.
     *
     * @param id  The id of the pci device
     * @return  the PCI object struct.
     */
    const io_obj_info* get_pci_device(size_type id) const
    {
        GKO_ENSURE_IN_BOUNDS(id, this->pci_devices_.size());
        return &this->pci_devices_[id];
    }

    /**
     * Get the object of type pci device associated with the PCI bus id.
     *
     * @param pci_bus_id  The PCI bus id of the pci device
     * @return  the PCI object struct.
     */
    const io_obj_info* get_pci_device(const std::string& pci_bus_id) const;

    /**
     * Get the number of PU objects stored in this Topology tree.
     *
     * @return  the number of PUs.
     */
    size_type get_num_pus() const { return this->pus_.size(); }

    /**
     * Get the number of core objects stored in this Topology tree.
     *
     * @return  the number of cores.
     */
    size_type get_num_cores() const { return this->cores_.size(); }

    /**
     * Get the number of PCI device objects stored in this Topology tree.
     *
     * @return  the number of PCI devices.
     */
    size_type get_num_pci_devices() const { return this->pci_devices_.size(); }

    /**
     * Get the number of NUMA objects stored in this Topology tree.
     *
     * @return  the number of NUMA objects.
     */
    size_type get_num_numas() const { return this->num_numas_; }

    /**
     * @internal
     *
     * A helper function that binds the calling process with the ids of `obj`
     * object .
     */
    void hwloc_binding_helper(
        const std::vector<machine_topology::normal_obj_info>& obj,
        const std::vector<int>& ids, const bool singlify = true) const;

    /**
     * @internal
     *
     * Load the objects of a normal HWLOC type (Packages, cores, numa-nodes).
     *
     * @note The objects should be sorted by logical index since hwloc uses
     * logical index with these functions
     */
    void load_objects(hwloc_obj_type_t type,
                      std::vector<normal_obj_info>& objects) const;

    /**
     * @internal
     *
     * Load the objects of io type (PCI devices and OS devices).
     *
     * @note The objects should be sorted by logical index since hwloc uses
     * logical index with these functions
     */
    void load_objects(hwloc_obj_type_t type,
                      std::vector<io_obj_info>& vector) const;

    /**
     *
     * @internal
     *
     * Get object id from the os index
     */
    int get_obj_id_by_os_index(const std::vector<normal_obj_info>& objects,
                               size_type os_index) const;

    /**
     *
     * @internal
     *
     * Get object id from the hwloc index
     */
    int get_obj_id_by_gp_index(const std::vector<normal_obj_info>& objects,
                               size_type gp_index) const;

private:
    /**
     * Do not allow the machine_topology object to be copied/moved. There should
     * be only one global object per execution.
     */
    machine_topology();
    machine_topology(machine_topology&) = delete;
    machine_topology(machine_topology&&) = delete;
    machine_topology& operator=(machine_topology&) = delete;
    machine_topology& operator=(machine_topology&&) = delete;
    ~machine_topology() = default;

    std::vector<normal_obj_info> pus_;
    std::vector<normal_obj_info> cores_;
    std::vector<normal_obj_info> packages_;
    std::vector<normal_obj_info> numa_nodes_;
    std::vector<io_obj_info> pci_devices_;
    size_type num_numas_;

    hwloc_manager<hwloc_topology> topo_;
};


using MachineTopology GKO_DEPRECATED("please use machine_topology") =
    machine_topology;


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_MACHINE_TOPOLOGY_HPP_
