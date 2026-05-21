#!/usr/bin/env julia
# Plot residual history for the randomized-gmres example.
#
# Usage:
#   julia plot_residuals.jl <csv_file> [output_image]
#
# CSV is the file produced by --residual-csv (long format:
# variant,iter,residual_norm). Default output: <csv_basename>.png next to
# the CSV.
#
# Requirements: Pkg.add(["DataFrames", "CSV", "Plots"])

using CSV
using DataFrames
using Plots
using Printf

if length(ARGS) < 1
    println(stderr, "usage: julia plot_residuals.jl <csv_file> [output_image]")
    exit(1)
end

csv_path = ARGS[1]
out_path = length(ARGS) >= 2 ? ARGS[2] :
           replace(csv_path, r"\.csv$" => ".png")

df = CSV.read(csv_path, DataFrame)

# Stable variant order so the legend matches the example's output.
order = ["mgs", "cgs", "cgs2", "rgs"]
present = filter(v -> v in df.variant, order)

# Distinct colors that are also colorblind-friendly enough at a glance.
palette = Dict(
    "mgs"  => :steelblue,
    "cgs"  => :darkorange,
    "cgs2" => :seagreen,
    "rgs"  => :crimson,
)

plt = plot(
    xlabel = "iteration",
    ylabel = "‖r‖₂",
    yscale = :log10,
    legend = :topright,
    title = "GMRES residual history",
    framestyle = :box,
    size = (900, 550),
    dpi = 150,
)

for v in present
    sub = df[df.variant .== v, :]
    final = last(sub.residual_norm)
    label = @sprintf("%s  (%d iters, final ‖r‖=%.2e)",
                     v, nrow(sub), final)
    plot!(plt, sub.iter, sub.residual_norm;
          label, lw = 2,
          color = get(palette, v, :black))
end

savefig(plt, out_path)
println("wrote $out_path")
