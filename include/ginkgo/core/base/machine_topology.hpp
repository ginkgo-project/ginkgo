// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_MACHINE_TOPOLOGY_HPP_
#define GKO_PUBLIC_CORE_BASE_MACHINE_TOPOLOGY_HPP_


#include <ginkgo/core/base/exception_helpers.hpp>


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


namespace gko {

class GKO_DEPRECATED("this class has no effect") machine_topology {
    struct GKO_DEPRECATED("") normal_obj_info {};

    struct GKO_DEPRECATED("") io_obj_info {};

public:
    static machine_topology* get_instance()
    {
        static machine_topology instance;
        return &instance;
    }

    void bind_to_cores(const std::vector<int>&, const bool) const {}

    void bind_to_core(const int&) const {}

    void bind_to_pus(const std::vector<int>&, const bool) const {}

    void bind_to_pu(const int&) const {}

    const normal_obj_info* get_pu(size_type) const { return nullptr; }

    const normal_obj_info* get_core(size_type) const { return nullptr; }

    const io_obj_info* get_pci_device(size_type) const { return nullptr; }

    const io_obj_info* get_pci_device(const std::string&) const
    {
        return nullptr;
    }

    size_type get_num_pus() const { return 0; }

    size_type get_num_cores() const { return 0; }

    size_type get_num_pci_devices() const { return 0; }

    size_type get_num_numas() const { return 0; }

private:
    machine_topology();
    machine_topology(machine_topology&) = delete;
    machine_topology(machine_topology&&) = delete;
    machine_topology& operator=(machine_topology&) = delete;
    machine_topology& operator=(machine_topology&&) = delete;
    ~machine_topology() = default;
};


using MachineTopology GKO_DEPRECATED("please use machine_topology") =
    machine_topology;


}  // namespace gko


GKO_END_DISABLE_DEPRECATION_WARNINGS


#endif  // GKO_PUBLIC_CORE_BASE_MACHINE_TOPOLOGY_HPP_
