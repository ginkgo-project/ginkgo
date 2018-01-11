#ifndef GKO_CORE_BASE_UTILS_HPP_
#define GKO_CORE_BASE_UTILS_HPP_


#include "core/base/exception_helpers.hpp"


namespace gko {


/**
 * Performs polymorphic type conversion.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the object which should be converted
 *
 * @return If successful, returns a pointer to the subtype, otherwise throws
 *         NotSupported.
 */
template <typename T, typename U>
T *as(U *obj)
{
    if (auto p = dynamic_cast<T *>(obj)) {
        return p;
    } else {
        throw NOT_SUPPORTED(obj);
    }
}

/**
 * Performs polymorphic type conversion.
 *
 * This is the constant version of the function.
 *
 * @tparam T  requested result type
 * @tparam U  static type of the passed object
 *
 * @param obj  the object which should be converted
 *
 * @return If successful, returns a pointer to the subtype, otherwise throws
 *         NotSupported.
 */
template <typename T, typename U>
const T *as(const U *obj)
{
    if (auto p = dynamic_cast<const T *>(obj)) {
        return p;
    } else {
        throw NOT_SUPPORTED(obj);
    }
}


}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
