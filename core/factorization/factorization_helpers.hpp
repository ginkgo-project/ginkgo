// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_CORE_FACTORIZATION_FACTORIZATION_HELPERS_HPP
#define GINKGO_CORE_FACTORIZATION_FACTORIZATION_HELPERS_HPP


#include <utility>


namespace gko {
namespace factorization {


struct identity {
    template <typename T>
    constexpr T operator()(T value)
    {
        return value;
    }
};


template <typename DiagClosure, typename OffDiagClosure>
class triangular_mtx_closure {
public:
    constexpr triangular_mtx_closure(DiagClosure diag_closure,
                                     OffDiagClosure off_diag_closure)
        : diag_closure_(std::move(diag_closure)),
          off_diag_closure_(std::move(off_diag_closure))
    {}

    template <typename T>
    constexpr T map_diag(T value)
    {
        return diag_closure_(value);
    }

    template <typename T>
    constexpr T map_off_diag(T value)
    {
        return off_diag_closure_(value);
    }

private:
    DiagClosure diag_closure_;
    OffDiagClosure off_diag_closure_;
};


}  // namespace factorization
}  // namespace gko


#endif  // GINKGO_CORE_FACTORIZATION_FACTORIZATION_HELPERS_HPP
