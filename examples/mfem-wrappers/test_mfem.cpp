//                                MFEM Example 1, modified
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"

#include "mfem_wrapper.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    const char *mesh_file = "beam-hex.mesh";
    int order = 2;
    bool static_cond = false;
    bool pa = true;
    const char *device_config = "cpu";
    bool visualization = true;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) or -1 for"
                   " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                   "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good()) {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    // 2. Enable hardware devices such as GPUs, and programming models such as
    //    CUDA, OCCA, RAJA and OpenMP based on command line options.
    Device device(device_config);
    device.Print();

    // 3. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
    //    the same code.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 50,000
    //    elements.
    {
        int ref_levels =
            (int)floor(log(50000. / mesh->GetNE()) / log(2.) / dim);
        for (int l = 0; l < ref_levels; l++) {
            mesh->UniformRefinement();
        }
    }

    // 5. Define a finite element space on the mesh. Here we use continuous
    //    Lagrange finite elements of the specified order. If order < 1, we
    //    instead use an isoparametric/isogeometric space.
    FiniteElementCollection *fec;
    if (order > 0) {
        fec = new H1_FECollection(order, dim);
    } else if (mesh->GetNodes()) {
        fec = mesh->GetNodes()->OwnFEC();
        cout << "Using isoparametric FEs: " << fec->Name() << endl;
    } else {
        fec = new H1_FECollection(order = 1, dim);
    }
    FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
         << endl;

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking all
    //    the boundary attributes from the mesh as essential (Dirichlet) and
    //    converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (mesh->bdr_attributes.Size()) {
        Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // 7. Set up the linear form b(.) which corresponds to the right-hand side
    // of
    //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
    //    the basis functions in the finite element fespace.
    LinearForm *b = new LinearForm(fespace);

    ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new DomainLFIntegrator(one));
    b->Assemble();

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    GridFunction x(fespace);
    x = 0.0;

    // 9. Set up the bilinear form a(.,.) on the finite element space
    //    corresponding to the Laplacian operator -Delta, by adding the
    //    Diffusion domain integrator.
    BilinearForm *a = new BilinearForm(fespace);
    if (pa) {
        a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    }
    a->AddDomainIntegrator(new DiffusionIntegrator(one));

    // 10. Assemble the bilinear form and the corresponding linear system,
    //     applying any necessary transformations such as: eliminating boundary
    //     conditions, applying conforming constraints for non-conforming AMR,
    //     static condensation, etc.
    if (static_cond) {
        a->EnableStaticCondensation();
    }
    a->Assemble();

    OperatorPtr A;
    Vector B, X;

    // TODO: is this addition needed now?
    X.UseDevice(true);

    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

    // 11. Solve the linear system A X = B.

    // ---------------------------------------------------------------
    // -------------------- Start Ginkgo section ---------------------

    // Ginkgo executor
    auto omp_executor = gko::OmpExecutor::create();
    auto cuda_executor =
        gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    std::shared_ptr<gko::Executor> executor;
    if (!strcmp(device_config, "cuda")) {
        executor = cuda_executor;
    } else {
        executor = omp_executor;
    }

    // MFEM Vector wrappers
    // The "true" here is now for on_device (ownership is default = false)
    auto gko_rhs = MFEMVectorWrapper::create(executor, B.Size(), &B, true);
    auto gko_x = MFEMVectorWrapper::create(executor, X.Size(), &X, true);

    // MFEM Operator Wrapper
    // Note we must set this to false or else the operator might be accidentally
    // deleted when doing a shallow copy to set mfem_oper_ in the
    // MFEMOperatorWrapper object.  The ownership
    //  only seems to be transferred for the PA case (and not full assembly)?
    A.SetOperatorOwner(false);
    auto oper_wrapper = MFEMOperatorWrapper::create(executor, B.Size(), A);

    // Create Ginkgo solver
    using cg = gko::solver::Cg<double>;

    // Generate solver
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(X.Size()).on(
                    executor),
                gko::stop::ResidualNormReduction<>::build().on(executor))
            .on(executor);
    auto solver = solver_gen->generate(gko::give(oper_wrapper));

    // Solve system
    solver->apply(gko::lend(gko_rhs), gko::lend(gko_x));

    // --------------------- End Ginkgo section ----------------------
    // ---------------------------------------------------------------

    // 12. Recover the solution as a finite element grid function.
    a->RecoverFEMSolution(X, *b, x);

    // 13. Save the refined mesh and the solution. This output can be viewed
    // later
    //     using GLVis: "glvis -m refined.mesh -g sol.gf".
    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    x.Save(sol_ofs);

    // 14. Send the solution by socket to a GLVis server.
    if (visualization) {
        char vishost[] = "localhost";
        int visport = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << x << flush;
    }

    //** TEST
    gko_x.release();

    // 15. Free the used memory.
    delete a;
    delete b;
    delete fespace;
    if (order > 0) {
        delete fec;
    }
    delete mesh;

    std::cout << "about to return 0 " << std::endl;

    return 0;
}
