#ifndef GKO_CORE_BASE_COPY_ASSIGNABLE_HPP
#define GKO_CORE_BASE_COPY_ASSIGNABLE_HPP


#include <vector>


namespace gko {
namespace detail {


template <typename T, typename = void>
class copy_assignable;


/**
 * Helper class to make a type copy assignable.
 *
 * This class wraps an object of a type that has a copy constructor, but not
 * a copy assignment. This is most often the case for lambdas. The wrapped
 * object can then be copy assigned, by relying on the copy constructor.
 *
 * @tparam T  type with a copy constructor
 */
template <typename T>
class copy_assignable<
    T, typename std::enable_if<std::is_copy_constructible<T>::value>::type> {
public:
    copy_assignable() : obj_{{}} {}
    copy_assignable(const copy_assignable& other) = default;
    copy_assignable(copy_assignable&& other) noexcept = default;

    copy_assignable(const T& obj) : obj_{obj} {}
    copy_assignable(T&& obj) : obj_{std::move(obj)} {}

    copy_assignable& operator=(const copy_assignable& other)
    {
        obj_.clear();
        obj_.emplace_back(other.get());
        return *this;
    }
    copy_assignable& operator=(copy_assignable&& other) noexcept = default;

    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const
    {
        return obj_[0](std::forward<Args>(args)...);
    }

    T const& get() const { return obj_[0]; }
    T& get() { return obj_[0]; }

private:
    //!< Store wrapped object in a container that has an emplace function
    std::vector<T> obj_;
};


}  // namespace detail
}  // namespace gko

#endif  // GKO_CORE_BASE_COPY_ASSIGNABLE_HPP
