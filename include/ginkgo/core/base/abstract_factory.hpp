// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_ABSTRACT_FACTORY_HPP_
#define GKO_PUBLIC_CORE_BASE_ABSTRACT_FACTORY_HPP_


#include <unordered_map>


#include <ginkgo/core/base/polymorphic_object.hpp>


/**
 * @brief The Ginkgo namespace.
 *
 * @ingroup gko
 */
namespace gko {


/**
 * The AbstractFactory is a generic interface template that enables easy
 * implementation of the abstract factory design pattern.
 *
 * The interface provides the AbstractFactory::generate() method that can
 * produce products of type `AbstractProductType` using an object of
 * `ComponentsType` (which can be constructed on the fly from parameters to its
 * constructors).
 * The generate() method is not declared as virtual, as this allows subclasses
 * to hide the method with a variant that preserves the compile-time type of the
 * objects. Instead, implementers should override the generate_impl() method,
 * which is declared virtual.
 *
 * Implementers of concrete factories should consider using the
 * EnableDefaultFactory mixin to obtain default implementations of utility
 * methods of PolymorphicObject and AbstractFactory.
 *
 * @tparam AbstractProductType  the type of products the factory produces
 * @tparam ComponentsType  the type of components the factory needs to produce
 *                         the product
 */
template <typename AbstractProductType, typename ComponentsType>
class AbstractFactory
    : public EnableAbstractPolymorphicObject<
          AbstractFactory<AbstractProductType, ComponentsType>> {
public:
    using abstract_product_type = AbstractProductType;
    using components_type = ComponentsType;

    /**
     * Creates a new product from the given components.
     *
     * The method will create an ComponentsType object from the arguments of
     * this method, and pass it to the generate_impl() function which will
     * create a new AbstractProductType.
     *
     * @tparam Args  types of arguments passed to the constructor of
     *               ComponentsType
     *
     * @param args  arguments passed to the constructor of ComponentsType
     *
     * @return an instance of AbstractProductType
     */
    template <typename... Args>
    std::unique_ptr<abstract_product_type> generate(Args&&... args) const
    {
        auto product =
            this->generate_impl(components_type{std::forward<Args>(args)...});
        for (auto logger : this->loggers_) {
            product->add_logger(logger);
        }
        return product;
    }

protected:
    /**
     * Constructs a new factory on the specified executor.
     *
     * @param exec  the executor where the factory should be constructed
     */
    AbstractFactory(std::shared_ptr<const Executor> exec)
        : EnableAbstractPolymorphicObject<AbstractFactory>(std::move(exec))
    {}

    /**
     * Constructs a new product from the given components.
     *
     * @param args  the components from which to create the product
     *
     * @return an instance of AbstractProductType
     */
    virtual std::unique_ptr<abstract_product_type> generate_impl(
        ComponentsType args) const = 0;
};


/**
 * This mixin provides a default implementation of a concrete factory.
 *
 * It implements all the methods of AbstractFactory and PolymorphicObject.
 * Its implementation of the generate_impl() method delegates the creation of
 * the product by calling the
 * `ProductType::ProductType(const ConcreteFactory *, const components_type &)`
 * constructor. The factory also supports parameters by using the
 * `ParametersType` structure, which is defined by the user.
 *
 * For a simple example, see IntFactory in
 * `core/test/base/abstract_factory.cpp`.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parameter]
 * @tparam ProductType  the concrete type of products which this factory
 *                      produces, has to be a subclass of
 *                      PolymorphicBase::abstract_product_type
 * @tparam ParametersType  a type representing the parameters of the factory,
 *                         has to inherit from the enable_parameters_type mixin
 * @tparam PolymorphicBase  parent of ConcreteFactory in the polymorphic
 *                          hierarchy, has to be a subclass of AbstractFactory
 */
template <typename ConcreteFactory, typename ProductType,
          typename ParametersType, typename PolymorphicBase>
class EnableDefaultFactory
    : public EnablePolymorphicObject<ConcreteFactory, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteFactory> {
public:
    friend class EnablePolymorphicObject<ConcreteFactory, PolymorphicBase>;

    using product_type = ProductType;
    using parameters_type = ParametersType;
    using polymorphic_base = PolymorphicBase;
    using abstract_product_type =
        typename PolymorphicBase::abstract_product_type;
    using components_type = typename PolymorphicBase::components_type;

    template <typename... Args>
    std::unique_ptr<product_type> generate(Args&&... args) const
    {
        auto product = std::unique_ptr<product_type>(static_cast<product_type*>(
            this->polymorphic_base::generate(std::forward<Args>(args)...)
                .release()));
        return product;
    }

    /**
     * Returns the parameters of the factory.
     *
     * @return the parameters of the factory
     */
    const parameters_type& get_parameters() const noexcept
    {
        return parameters_;
    };

    /**
     * Creates a new ParametersType object which can be used to instantiate a
     * new ConcreteFactory.
     *
     * This method does not construct the factory directly, but returns a new
     * parameters_type object, which can be used to set the parameters of the
     * factory. Once the parameters have been set, the
     * parameters_type::on() method can be used to obtain an instance
     * of the factory with those parameters.
     *
     * @return a default parameters_type object
     */
    static parameters_type create() { return {}; }

protected:
    /**
     * Creates a new factory using the specified executor and parameters.
     *
     * @param exec  the executor where the factory will be constructed
     * @param parameters  the parameters structure for the factory
     */
    explicit EnableDefaultFactory(std::shared_ptr<const Executor> exec,
                                  const parameters_type& parameters = {})
        : EnablePolymorphicObject<ConcreteFactory, PolymorphicBase>(
              std::move(exec)),
          parameters_{parameters}
    {}

    std::unique_ptr<abstract_product_type> generate_impl(
        components_type args) const override
    {
        return std::unique_ptr<abstract_product_type>(
            new product_type(self(), args));
    }

private:
    GKO_ENABLE_SELF(ConcreteFactory);

    ParametersType parameters_;
};


/**
 * The enable_parameters_type mixin is used to create a base implementation of
 * the factory parameters structure.
 *
 * It provides only the on() method which can be used to instantiate
 * the factory give the parameters stored in the structure.
 *
 * @tparam ConcreteParametersType  the concrete parameters type which is being
 *                                 implemented [CRTP parameter]
 * @tparam Factory  the concrete factory for which these parameters are being
 *                  used
 */
template <typename ConcreteParametersType, typename Factory>
class enable_parameters_type {
public:
    using factory = Factory;

    /**
     * Provides the loggers to be added to the factory and its generated
     * objects in a fluent interface.
     */
    template <typename... Args>
    ConcreteParametersType& with_loggers(Args&&... _value)
    {
        this->loggers = {std::forward<Args>(_value)...};
        return *self();
    }

    /**
     * Creates a new factory on the specified executor.
     *
     * @param exec  the executor where the factory will be created
     *
     * @return a new factory instance
     */
    std::unique_ptr<Factory> on(std::shared_ptr<const Executor> exec) const
    {
        ConcreteParametersType copy = *self();
        for (const auto& item : deferred_factories) {
            item.second(exec, copy);
        }
        auto factory = std::unique_ptr<Factory>(new Factory(exec, copy));
        for (auto& logger : loggers) {
            factory->add_logger(logger);
        };
        return factory;
    }

protected:
    GKO_ENABLE_SELF(ConcreteParametersType);

    /**
     * Loggers to be attached to the factory and generated object.
     */
    std::vector<std::shared_ptr<const log::Logger>> loggers{};

    /**
     * Deferred factory parameter initialization functions that will be called
     * in on(). Their names usually correspond to the variable names in the
     * parameter type. They will be provided the executor and the parameter
     * object currently being initialized from the generators.
     */
    std::unordered_map<std::string,
                       std::function<void(std::shared_ptr<const Executor> exec,
                                          ConcreteParametersType&)>>
        deferred_factories;
};


/**
 * This Macro will generate a new type containing the parameters for the factory
 * `_factory_name`. For more details, see #GKO_ENABLE_LIN_OP_FACTORY().
 * It is required to use this macro **before** calling the
 * macro #GKO_ENABLE_LIN_OP_FACTORY().
 * It is also required to use the same names for all parameters between both
 * macros.
 *
 * @param _parameters_name  name of the parameters member in the class
 * @param _factory_name  name of the generated factory type
 *
 * @ingroup LinOp
 */
#define GKO_CREATE_FACTORY_PARAMETERS(_parameters_name, _factory_name)  \
public:                                                                 \
    class _factory_name;                                                \
    struct _parameters_name##_type                                      \
        : public ::gko::enable_parameters_type<_parameters_name##_type, \
                                               _factory_name>


namespace detail {


// Use pointer not the type because std::is_convertible<const type, type> can be
// true.
template <typename From, typename To>
struct is_pointer_convertible : std::is_convertible<From*, To*> {};


}  // namespace detail


/**
 * Represents a factory parameter of factory type that can either initialized by
 * a pre-existing factory or by passing in a factory_parameters object whose
 * `.on(exec)` will be called to instantiate a factory.
 *
 * @tparam FactoryType  the type of factory that can be instantiated from this
 * object.
 */
template <typename FactoryType>
class deferred_factory_parameter {
public:
    /** Creates an empty deferred factory parameter. */
    deferred_factory_parameter() = default;

    /** Creates a deferred factory parameter returning a nullptr. */
    deferred_factory_parameter(std::nullptr_t)
    {
        generator_ = [](std::shared_ptr<const Executor>) { return nullptr; };
    }

    /**
     * Creates a deferred factory parameter from a preexisting factory with
     * shared ownership.
     */
    template <typename ConcreteFactoryType,
              std::enable_if_t<detail::is_pointer_convertible<
                  ConcreteFactoryType, FactoryType>::value>* = nullptr>
    deferred_factory_parameter(std::shared_ptr<ConcreteFactoryType> factory)
    {
        generator_ = [factory =
                          std::shared_ptr<FactoryType>(std::move(factory))](
                         std::shared_ptr<const Executor>) { return factory; };
    }

    /**
     * Creates a deferred factory parameter by taking ownership of a
     * preexisting factory with unique ownership.
     */
    template <typename ConcreteFactoryType, typename Deleter,
              std::enable_if_t<detail::is_pointer_convertible<
                  ConcreteFactoryType, FactoryType>::value>* = nullptr>
    deferred_factory_parameter(
        std::unique_ptr<ConcreteFactoryType, Deleter> factory)
    {
        generator_ = [factory =
                          std::shared_ptr<FactoryType>(std::move(factory))](
                         std::shared_ptr<const Executor>) { return factory; };
    }

    /**
     * Creates a deferred factory parameter object from a
     * factory_parameters-like object. To instantiate the actual factory, the
     * parameter's `.on(exec)` function will be called.
     */
    template <typename ParametersType,
              typename U = decltype(std::declval<ParametersType>().on(
                  std::shared_ptr<const Executor>{})),
              std::enable_if_t<detail::is_pointer_convertible<
                  typename U::element_type, FactoryType>::value>* = nullptr>
    deferred_factory_parameter(ParametersType parameters)
    {
        generator_ = [parameters](std::shared_ptr<const Executor> exec)
            -> std::shared_ptr<FactoryType> { return parameters.on(exec); };
    }

    /**
     * Instantiates the deferred parameter into an actual factory. This will
     * throw if the deferred factory parameter is empty.
     */
    std::shared_ptr<FactoryType> on(std::shared_ptr<const Executor> exec) const
    {
        if (this->is_empty()) {
            GKO_NOT_SUPPORTED(*this);
        }
        return generator_(exec);
    }

    /** Returns true iff the parameter is empty. */
    bool is_empty() const { return !bool(generator_); }

private:
    std::function<std::shared_ptr<FactoryType>(std::shared_ptr<const Executor>)>
        generator_;
};


/**
 * Defines a build method for the factory, simplifying its construction by
 * removing the repetitive typing of factory's name.
 *
 * @param _factory_name  the factory for which to define the method
 *
 * @ingroup LinOp
 */
#define GKO_ENABLE_BUILD_METHOD(_factory_name)                               \
    static auto build()->decltype(_factory_name::create())                   \
    {                                                                        \
        return _factory_name::create();                                      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#if !(defined(__CUDACC__) || defined(__HIPCC__))
/**
 * Creates a factory parameter in the factory parameters structure.
 *
 * @param _name  name of the parameter
 * @param __VA_ARGS__  default value of the parameter
 *
 * @see GKO_ENABLE_LIN_OP_FACTORY for more details, and usage example
 *
 * @deprecated Use GKO_FACTORY_PARAMETER_SCALAR or GKO_FACTORY_PARAMETER_VECTOR
 *
 * @ingroup LinOp
 */
#define GKO_FACTORY_PARAMETER(_name, ...)                                    \
    _name{__VA_ARGS__};                                                      \
                                                                             \
    template <typename... Args>                                              \
    auto with_##_name(Args&&... _value)                                      \
        ->std::decay_t<decltype(*(this->self()))>&                           \
    {                                                                        \
        using type = decltype(this->_name);                                  \
        this->_name = type{std::forward<Args>(_value)...};                   \
        return *(this->self());                                              \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

/**
 * Creates a scalar factory parameter in the factory parameters structure.
 *
 * Scalar in this context means that the constructor for this type only takes
 * a single parameter.
 *
 * @param _name  name of the parameter
 * @param _default  default value of the parameter
 *
 * @see GKO_ENABLE_LIN_OP_FACTORY for more details, and usage example
 *
 * @ingroup LinOp
 */
#define GKO_FACTORY_PARAMETER_SCALAR(_name, _default) \
    GKO_FACTORY_PARAMETER(_name, _default)

/**
 * Creates a vector factory parameter in the factory parameters structure.
 *
 * Vector in this context means that the constructor for this type takes
 * multiple parameters.
 *
 * @param _name  name of the parameter
 * @param _default  default value of the parameter
 *
 * @see GKO_ENABLE_LIN_OP_FACTORY for more details, and usage example
 *
 * @ingroup LinOp
 */
#define GKO_FACTORY_PARAMETER_VECTOR(_name, ...) \
    GKO_FACTORY_PARAMETER(_name, __VA_ARGS__)
#else  // defined(__CUDACC__) || defined(__HIPCC__)
// A workaround for the NVCC compiler - parameter pack expansion does not work
// properly, because while the assignment to a scalar value is translated by
// cudafe into a C-style cast, the parameter pack expansion is not removed and
// `Args&&... args` is still kept as a parameter pack.
#define GKO_FACTORY_PARAMETER(_name, ...)                                    \
    _name{__VA_ARGS__};                                                      \
                                                                             \
    template <typename... Args>                                              \
    auto with_##_name(Args&&... _value)                                      \
        ->std::decay_t<decltype(*(this->self()))>&                           \
    {                                                                        \
        GKO_NOT_IMPLEMENTED;                                                 \
        return *(this->self());                                              \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_FACTORY_PARAMETER_SCALAR(_name, _default)                         \
    _name{_default};                                                          \
                                                                              \
    template <typename Arg>                                                   \
    auto with_##_name(Arg&& _value)->std::decay_t<decltype(*(this->self()))>& \
    {                                                                         \
        using type = decltype(this->_name);                                   \
        this->_name = type{std::forward<Arg>(_value)};                        \
        return *(this->self());                                               \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")

#define GKO_FACTORY_PARAMETER_VECTOR(_name, ...)                             \
    _name{__VA_ARGS__};                                                      \
                                                                             \
    template <typename... Args>                                              \
    auto with_##_name(Args&&... _value)                                      \
        ->std::decay_t<decltype(*(this->self()))>&                           \
    {                                                                        \
        using type = decltype(this->_name);                                  \
        this->_name = type{std::forward<Args>(_value)...};                   \
        return *(this->self());                                              \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif  // defined(__CUDACC__) || defined(__HIPCC__)

/**
 * Creates a factory parameter of factory type. The parameter can either be set
 * directly, or its creation can be deferred until the executor is set in the
 * `.on(exec)` function call, by using a deferred_factory_parameter.
 *
 * @param _name  name of the parameter
 * @param _type  pointee type of the parameter, e.g. LinOpFactory
 *
 */
#define GKO_DEFERRED_FACTORY_PARAMETER(_name)                                  \
    _name{};                                                                   \
                                                                               \
private:                                                                       \
    using _name##_type = typename decltype(_name)::element_type;               \
                                                                               \
public:                                                                        \
    auto with_##_name(::gko::deferred_factory_parameter<_name##_type> factory) \
        ->std::decay_t<decltype(*(this->self()))>&                             \
    {                                                                          \
        this->_name##_generator_ = std::move(factory);                         \
        this->deferred_factories[#_name] = [](const auto& exec,                \
                                              auto& params) {                  \
            if (!params._name##_generator_.is_empty()) {                       \
                params._name = params._name##_generator_.on(exec);             \
            }                                                                  \
        };                                                                     \
        return *(this->self());                                                \
    }                                                                          \
                                                                               \
private:                                                                       \
    ::gko::deferred_factory_parameter<_name##_type> _name##_generator_;        \
                                                                               \
public:                                                                        \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")

/**
 * Creates a factory parameter representing a vector of factories type. The
 * parameter can either be set directly, or its creation can be deferred until
 * the executor is set in the
 * `.on(exec)` function call, by using a vector of deferred_factory_parameters.
 *
 * @param _name  name of the parameter
 * @param _type  pointee type of the vector entries, e.g. LinOpFactory
 *
 */
#define GKO_DEFERRED_FACTORY_VECTOR_PARAMETER(_name)                           \
    _name{};                                                                   \
                                                                               \
private:                                                                       \
    using _name##_type = typename decltype(_name)::value_type::element_type;   \
                                                                               \
public:                                                                        \
    template <typename... Args,                                                \
              typename = std::enable_if_t<::gko::xstd::conjunction<            \
                  std::is_convertible<Args, ::gko::deferred_factory_parameter< \
                                                _name##_type>>...>::value>>    \
    auto with_##_name(Args&&... factories)                                     \
        ->std::decay_t<decltype(*(this->self()))>&                             \
    {                                                                          \
        this->_name##_generator_ = {                                           \
            ::gko::deferred_factory_parameter<_name##_type>{                   \
                std::forward<Args>(factories)}...};                            \
        this->deferred_factories[#_name] = [](const auto& exec,                \
                                              auto& params) {                  \
            if (!params._name##_generator_.empty()) {                          \
                params._name.clear();                                          \
                for (auto& generator : params._name##_generator_) {            \
                    params._name.push_back(generator.on(exec));                \
                }                                                              \
            }                                                                  \
        };                                                                     \
        return *(this->self());                                                \
    }                                                                          \
    template <typename FactoryType,                                            \
              typename = std::enable_if_t<std::is_convertible<                 \
                  FactoryType,                                                 \
                  ::gko::deferred_factory_parameter<_name##_type>>::value>>    \
    auto with_##_name(const std::vector<FactoryType>& factories)               \
        ->std::decay_t<decltype(*(this->self()))>&                             \
    {                                                                          \
        this->_name##_generator_.clear();                                      \
        for (const auto& factory : factories) {                                \
            this->_name##_generator_.push_back(factory);                       \
        }                                                                      \
        this->deferred_factories[#_name] = [](const auto& exec,                \
                                              auto& params) {                  \
            if (!params._name##_generator_.empty()) {                          \
                params._name.clear();                                          \
                for (auto& generator : params._name##_generator_) {            \
                    params._name.push_back(generator.on(exec));                \
                }                                                              \
            }                                                                  \
        };                                                                     \
        return *(this->self());                                                \
    }                                                                          \
                                                                               \
private:                                                                       \
    std::vector<::gko::deferred_factory_parameter<_name##_type>>               \
        _name##_generator_;                                                    \
                                                                               \
public:                                                                        \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_ABSTRACT_FACTORY_HPP_
