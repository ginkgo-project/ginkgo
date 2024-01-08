// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>


/**
 * The ByInteraction class is a criterion which asks for user input to stop
 * the iteration process. Using this criterion is slightly more complex than the
 * other ones, because it is asynchronous therefore requires the use of threads.
 */
class ByInteraction
    : public gko::EnablePolymorphicObject<ByInteraction, gko::stop::Criterion> {
    friend class gko::EnablePolymorphicObject<ByInteraction,
                                              gko::stop::Criterion>;
    using Criterion = gko::stop::Criterion;

public:
    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Boolean set by the user to stop the iteration process
         */
        std::add_pointer<volatile bool>::type GKO_FACTORY_PARAMETER_SCALAR(
            stop_iteration_process, nullptr);
    };
    GKO_ENABLE_CRITERION_FACTORY(ByInteraction, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    bool check_impl(gko::uint8 stoppingId, bool setFinalized,
                    gko::array<gko::stopping_status>* stop_status,
                    bool* one_changed, const Criterion::Updater&) override
    {
        bool result = *(parameters_.stop_iteration_process);
        if (result) {
            this->set_all_statuses(stoppingId, setFinalized, stop_status);
            *one_changed = true;
        }
        return result;
    }

    explicit ByInteraction(std::shared_ptr<const gko::Executor> exec)
        : EnablePolymorphicObject<ByInteraction, Criterion>(std::move(exec))
    {}

    explicit ByInteraction(const Factory* factory,
                           const gko::stop::CriterionArgs& args)

        : EnablePolymorphicObject<ByInteraction, Criterion>(
              factory->get_executor()),
          parameters_{factory->get_parameters()}
    {}
};


void run_solver(volatile bool* stop_iteration_process,
                std::shared_ptr<gko::Executor> exec)
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using bicg = gko::solver::Bicgstab<ValueType>;

    // Read Data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    // Create solver factory and solve system
    auto solver =
        bicg::build()
            .with_criteria(ByInteraction::build().with_stop_iteration_process(
                stop_iteration_process))
            .on(exec)
            ->generate(A);
    solver->add_logger(gko::log::Stream<ValueType>::create(
        gko::log::Logger::iteration_complete_mask, std::cout, true));
    solver->apply(b, x);

    std::cout << "Solver stopped" << std::endl;

    // Print solution
    std::cout << "Solution (x): \n";
    write(std::cout, x);

    // Calculate residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Residual norm sqrt(r^T r): \n";
    write(std::cout, res);
}


int main(int argc, char* argv[])
{
    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Figure out where to run the code
    const auto executor_string = argc >= 2 ? argv[1] : "reference";

    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // Declare a user controlled boolean for the iteration process
    volatile bool stop_iteration_process{};

    // Create a new a thread to launch the solver
    std::thread t(run_solver, &stop_iteration_process, exec);

    // Look for an input command "stop" in the console, which sets the boolean
    // to true
    std::cout << "Type 'stop' to stop the iteration process" << std::endl;
    std::string command;
    while (std::cin >> command) {
        if (command == "stop") {
            break;
        } else {
            std::cout << "Unknown command" << std::endl;
        }
    }
    std::cout << "User input command 'stop' - The solver will stop!"
              << std::endl;
    stop_iteration_process = true;
    t.join();
}
