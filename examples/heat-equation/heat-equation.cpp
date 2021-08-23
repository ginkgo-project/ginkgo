/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

/*****************************<DESCRIPTION>***********************************
This example solves a 2D heat conduction equation

    u : [0, d]^2 \rightarrow R\\
    \partial_t u = \delta u + f

with Dirichlet boundary conditions and given initial condition and
constant-in-time source function f.

The partial differential equation (PDE) is solved with a finite difference
spatial discretization on an equidistant grid: For `n` grid points,
and grid distance $h = 1/n$ we write

    u_{i,j}' = \alpha (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1}
                   - 4 u_{i,j}) / h^2
               + f_{i,j}

We then build an implicit Euler integrator by discretizing with time step $\tau$

    (u_{i,j}^{k+1} - u_{i,j}^k) / \tau =
    \alpha (u_{i-1,j}^{k+1} - u_{i+1,j}^{k+1}
          + u_{i,j-1}^{k+1} - u_{i,j+1}^{k+1} - 4 u_{i,j}^{k+1}) / h^2
    + f_{i,j}

and solve the resulting linear system for $ u_{\cdot}^{k+1}$ using Ginkgo's CG
solver preconditioned with an incomplete Cholesky factorization for each time
step, occasionally writing the resulting grid values into a video file using
OpenCV and a custom color mapping.

The intention of this example is to provide a mini-app showing matrix assembly,
vector initialization, solver setup and the use of Ginkgo in a more complex
setting.
*****************************<DESCRIPTION>**********************************/

#include <ginkgo/ginkgo.hpp>


#include <chrono>
#include <fstream>
#include <iostream>


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>


// This function implements a simple Ginkgo-themed clamped color mapping for
// values in the range [0,5].
void set_val(unsigned char *data, double value)
{
    // RGB values for the 6 colors used for values 0, 1, ..., 5
    // We will interpolate linearly between these values.
    double col_r[] = {255, 221, 129, 201, 249, 255};
    double col_g[] = {255, 220, 130, 161, 158, 204};
    double col_b[] = {255, 220, 133, 93, 24, 8};
    value = std::max(0.0, value);
    auto i = std::max(0, std::min(4, int(value)));
    auto d = std::max(0.0, std::min(1.0, value - i));
    // OpenCV uses BGR instead of RGB by default, revert indices
    data[2] = static_cast<unsigned char>(col_r[i + 1] * d + col_r[i] * (1 - d));
    data[1] = static_cast<unsigned char>(col_g[i + 1] * d + col_g[i] * (1 - d));
    data[0] = static_cast<unsigned char>(col_b[i + 1] * d + col_b[i] * (1 - d));
}


// Initialize video output with given dimension and FPS (frames per seconds)
std::pair<cv::VideoWriter, cv::Mat> build_output(int n, double fps)
{
    cv::Size videosize{n, n};
    auto output =
        std::make_pair(cv::VideoWriter{}, cv::Mat{videosize, CV_8UC3});
    auto fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    output.first.open("heat.mp4", fourcc, fps, videosize);
    return output;
}


// Write the current frame to video output using the above color mapping
void output_timestep(std::pair<cv::VideoWriter, cv::Mat> &output, int n,
                     const double *data)
{
    for (int i = 0; i < n; i++) {
        auto row = output.second.ptr(i);
        for (int j = 0; j < n; j++) {
            set_val(&row[3 * j], data[i * n + j]);
        }
    }
    output.first.write(output.second);
}


int main(int argc, char *argv[])
{
    using mtx = gko::matrix::Csr<>;
    using vec = gko::matrix::Dense<>;

    // Problem parameters:
    // simulation length
    auto t0 = 5.0;
    // diffusion factor
    auto diffusion = 0.0005;
    // scaling factor for heat source
    auto source_scale = 2.5;
    // Simulation parameters:
    // inner grid points per discretization direction
    auto n = 256;
    // number of simulation steps per second
    auto steps_per_sec = 500;
    // number of video frames per second
    auto fps = 25;
    // number of grid points
    auto n2 = n * n;
    // grid point distance (ignoring boundary points)
    auto h = 1.0 / (n + 1);
    auto h2 = h * h;
    // time step size for the simulation
    auto tau = 1.0 / steps_per_sec;

    // create a CUDA executor with an associated OpenMP host executor
    auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    // load heat source and initial state vectors
    std::ifstream source_stream("data/source.mtx");
    std::ifstream initial_stream("data/initial.mtx");
    auto source = gko::read<vec>(source_stream, exec);
    auto in_vector = gko::read<vec>(initial_stream, exec);
    // create output vector with initial guess for
    auto out_vector = in_vector->clone();
    // create scalar for source update
    auto tau_source_scalar = gko::initialize<vec>({source_scale * tau}, exec);
    // create stencil matrix as shared_ptr for solver
    auto stencil_matrix = gko::share(mtx::create(exec));
    // assemble matrix
    gko::matrix_data<> mtx_data{gko::dim<2>(n2, n2)};
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            auto c = i * n + j;
            auto c_val = diffusion * tau * 4.0 / h2 + 1.0;
            auto off_val = -diffusion * tau / h2;
            // for each grid point: insert 5 stencil points
            // with Dirichlet boundary conditions, i.e. with zero boundary value
            if (i > 0) {
                mtx_data.nonzeros.emplace_back(c, c - n, off_val);
            }
            if (j > 0) {
                mtx_data.nonzeros.emplace_back(c, c - 1, off_val);
            }
            mtx_data.nonzeros.emplace_back(c, c, c_val);
            if (j < n - 1) {
                mtx_data.nonzeros.emplace_back(c, c + 1, off_val);
            }
            if (i < n - 1) {
                mtx_data.nonzeros.emplace_back(c, c + n, off_val);
            }
        }
    }
    stencil_matrix->read(mtx_data);
    // prepare video output
    auto output = build_output(n, fps);
    // build CG solver on stencil with incomplete Cholesky preconditioner
    // stopping at 1e-10 relative accuracy
    auto solver =
        gko::solver::Cg<>::build()
            .with_preconditioner(gko::preconditioner::Ic<>::build().on(exec))
            .with_criteria(gko::stop::RelativeResidualNorm<>::build()
                               .with_tolerance(1e-10)
                               .on(exec))
            .on(exec)
            ->generate(stencil_matrix);
    // time stamp of the last output frame (initialized to a sentinel value)
    double last_t = -t0;
    // execute implicit Euler method: for each timestep, solve stencil system
    for (double t = 0; t < t0; t += tau) {
        // if enough time has passed, output the next video frame
        if (t - last_t > 1.0 / fps) {
            last_t = t;
            std::cout << t << std::endl;
            output_timestep(
                output, n,
                gko::make_temporary_clone(exec->get_master(), in_vector.get())
                    ->get_const_values());
        }
        // add heat source contribution
        in_vector->add_scaled(gko::lend(tau_source_scalar), gko::lend(source));
        // execute Euler step
        solver->apply(gko::lend(in_vector), gko::lend(out_vector));
        // swap input and output
        std::swap(in_vector, out_vector);
    }
}
