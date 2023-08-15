// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_ABSTRACT_FACTORY_HPP_
#define GKO_PUBLIC_CORE_BASE_ABSTRACT_FACTORY_HPP_


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
        auto factory = std::unique_ptr<Factory>(new Factory(exec, *self()));
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
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_ABSTRACT_FACTORY_HPP_
