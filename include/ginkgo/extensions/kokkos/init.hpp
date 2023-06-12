#ifndef GINKGO_INIT_HPP
#define GINKGO_INIT_HPP

#include <ginkgo/core/base/executor.hpp>

#include <Kokkos_Core.hpp>


namespace gko {
namespace ext {
namespace kokkos {


Kokkos::ScopeGuard create_scope_guard(std::shared_ptr<const Executor> exec)
{
    auto settings = [&] {
        auto settings = Kokkos::InitializationSettings();
        if (auto p = dynamic_cast<const CudaExecutor*>(exec.get())) {
            return settings.set_device_id(p->get_device_id());
        }
        if (auto p = dynamic_cast<const HipExecutor*>(exec.get())) {
            return settings.set_device_id(p->get_device_id());
        }
        if (auto p = dynamic_cast<const DpcppExecutor*>(exec.get())) {
            return settings.set_device_id(p->get_device_id());
        }
        return settings;
    }();

    return {settings};
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_INIT_HPP
