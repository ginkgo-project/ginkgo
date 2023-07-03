#ifndef GKO_PUBLIC_CORE_BASE_NATIVE_TYPE_HPP_
#define GKO_PUBLIC_CORE_BASE_NATIVE_TYPE_HPP_

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace detail {


template <typename T, template <class, class...> class am,
          template <class, class...> class dm, class... Args>
struct native;


template <typename T, template <class, class...> class am,
          template <class, class...> class dm, class... Args>
struct native<array<T>, am, dm, Args...> : public am<T, Args...> {};

template <typename T, template <class, class...> class am,
          template <class, class...> class dm, class... Args>
struct native<const array<T>, am, dm, Args...> : public am<const T, Args...> {};


template <typename T, template <class, class...> class am,
          template <class, class...> class dm, class... Args>
struct native<matrix::Dense<T>, am, dm, Args...> : public dm<T, Args...> {};

template <typename T, template <class, class...> class am,
          template <class, class...> class dm, class... Args>
struct native<const matrix::Dense<T>, am, dm, Args...>
    : public dm<const T, Args...> {};


namespace impl {


template <typename ValueType, typename IndexType,
          template <class, class...> class array_mapper, typename... ExtraArgs>
struct device_matrix_data {
    using value_array = array_mapper<ValueType, ExtraArgs...>;
    using index_array = array_mapper<IndexType, ExtraArgs...>;

    struct type {
        typename index_array::type row_idxs;
        typename index_array::type col_idxs;
        typename value_array::type values;
    };

    static type map(std::shared_ptr<const Executor> exec, size_type num_elems,
                    IndexType* row_idxs, IndexType* col_idxs, ValueType* values)
    {
        return {
            index_array::map(gko::make_array_view(exec, num_elems, row_idxs)),
            index_array::map(gko::make_array_view(exec, num_elems, col_idxs)),
            value_array::map(gko::make_array_view(exec, num_elems, values))};
    }
};


}  // namespace impl


template <typename ValueType, typename IndexType,
          template <class, class...> class array_mapper,
          template <class, class...> class dm, typename... ExtraArgs>
struct native<device_matrix_data<ValueType, IndexType>, array_mapper, dm,
              ExtraArgs...> {
private:
    using impl = impl::device_matrix_data<ValueType, IndexType, array_mapper,
                                          ExtraArgs...>;

public:
    using type = typename impl::type;

    static type map(device_matrix_data<ValueType, IndexType>& md)
    {
        return impl::map(md.get_executor(), md.get_num_elems(),
                         md.get_row_idxs(), md.get_col_idxs(), md.get_values());
    }
};

template <typename ValueType, typename IndexType,
          template <class, class...> class array_mapper,
          template <class, class...> class dm, typename... ExtraArgs>
struct native<const device_matrix_data<ValueType, IndexType>, array_mapper, dm,
              ExtraArgs...> {
private:
    using impl = impl::device_matrix_data<ValueType, IndexType, array_mapper,
                                          ExtraArgs...>;

public:
    using type = typename impl::type;

    static type map(const device_matrix_data<ValueType, IndexType>& md)
    {
        return impl::map(md.get_executor(), md.get_num_elems(),
                         const_cast<IndexType>(md.get_const_row_idxs()),
                         const_cast<IndexType>(md.get_const_col_idxs()),
                         const_cast<ValueType>(md.get_values()));
    }
};


}  // namespace detail


template <template <class, class...> class array_mapper,
          template <class, class...> class dense_mapper, class... Args>
struct native {
private:
    template <typename T>
    using sanitize = typename std::remove_pointer_t<std::remove_reference_t<T>>;

    template <typename T>
    using native_impl = typename detail::native<sanitize<T>, array_mapper,
                                                dense_mapper, Args...>;

public:
    template <typename T>
    using type = typename native_impl<T>::type;

    template <typename T>
    static type<T> map(T&& input)
    {
        return native_impl<T>::map(input);
    }

    template <typename T>
    static type<T> map(T* input)
    {
        return map(*input);
    }

    template <typename T>
    static type<T> map(std::shared_ptr<T>& input)
    {
        return map(*input);
    }

    template <typename T>
    static type<T> map(std::unique_ptr<T>& input)
    {
        return map(*input);
    }
};

}  // namespace gko
#endif  // GKO_PUBLIC_CORE_BASE_NATIVE_TYPE_HPP_
