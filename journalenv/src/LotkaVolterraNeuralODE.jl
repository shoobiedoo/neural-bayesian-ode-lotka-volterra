# SciML Libraries
using SciMLSensitivity, DifferentialEquations

# ML Tools
using Lux, Zygote

# External Tools
using Random, Plots, AdvancedHMC, MCMCChains, StatsPlots, ComponentArrays

# Set random seed for reproducibility
Random.seed!(123)

# Define the system parameters
struct LVParameters
    α::Float64  # Prey death rate
    β::Float64  # Predation rate
    γ::Float64  # Predator growth rate
    δ::Float64  # Predator death rate
end

# Initialize parameters
lv_params = LVParameters(1.5, 1.0, 3.0, 1.0)

# Generate training data
function generate_training_data(params::LVParameters)
    u0 = [1.0; 1.0]  # Initial conditions
    datasize = 40
    tspan = (0.0, 10.0)
    tsteps = range(tspan[1], tspan[2], length = datasize)
    
    function lotka_volterra(du, u, p, t)
        du[1] = -params.α*u[1] - params.β*u[1]*u[2]
        du[2] = -params.δ*u[2] + params.γ*u[1]*u[2]
    end
    
    prob = ODEProblem(lotka_volterra, u0, tspan)
    Array(solve(prob, Tsit5(), saveat = tsteps))
end

# Generate the training data and constants
ode_data = generate_training_data(lv_params)
tspan = (0.0, 10.0)
u0 = [1.0; 1.0]
tsteps = range(tspan[1], tspan[2], length = 40)

# Define and create the neural network
dudt2 = Lux.Chain(x -> x .^ 3,
                  Lux.Dense(2, 50, tanh),
                  Lux.Dense(50, 2))

rng = Random.default_rng()
p, st = Lux.setup(rng, dudt2)
_st = st
_p = ComponentArray{Float64}(p)

# Neural ODE functions
function neural_ode_func(u, p, t)
    dudt2(u, p, _st)[1]
end

function prob_neural_ode(u0, p)
    prob = ODEProblem(neural_ode_func, u0, tspan, p)
    solve(prob, Tsit5(), saveat = tsteps)
end

# Prediction and loss functions
function predict_neural_ode(p)
    p = p isa ComponentArray ? p : convert(typeof(_p), p)
    Array(prob_neural_ode(u0, p))
end

function loss_neural_ode(p)
    pred = predict_neural_ode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Define the log probability function and its gradient
l(θ) = -sum(abs2, ode_data .- predict_neural_ode(θ)) - sum(θ .* θ)

function dldθ(θ)
    x, lambda = Zygote.pullback(l, θ)
    grad = first(lambda(1))
    return x, grad
end

# Setup HMC sampler
function setup_hmc(p)
    metric = DiagEuclideanMetric(length(p))
    h = Hamiltonian(metric, l, dldθ)
    integrator = Leapfrog(find_good_stepsize(h, p))
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), 
                            StepSizeAdaptor(0.45, integrator))
    return h, kernel, adaptor
end

# Run HMC sampling
function run_hmc_sampling(p, n_samples=500, n_adapts=500)
    h, kernel, adaptor = setup_hmc(p)
    samples, stats = sample(h, kernel, p, n_samples, adaptor, n_adapts; 
                          progress=true)
    return samples, stats
end

using Statistics
# Analysis functions
function plot_results(samples, ode_data)
    # Convert predictions to array format
    predictions = cat([predict_neural_ode(samples[i]) for i in 1:length(samples)]..., dims=3)
    
    # Calculate mean and std along the third dimension
    pred_mean = mean(predictions, dims=3)[:,:,1]
    pred_std = std(predictions, dims=3)[:,:,1]
    
    # Create the plot
    plt = plot(tsteps, ode_data[1,:], label="True Prey", color=:blue, 
              linewidth=2, xlabel="Time", ylabel="Population")
    plot!(tsteps, ode_data[2,:], label="True Predator", color=:red, linewidth=2)
    
    # Add predictions with uncertainty bands
    plot!(tsteps, pred_mean[1,:], ribbon=2pred_std[1,:], 
          label="Predicted Prey", color=:lightblue, alpha=0.3)
    plot!(tsteps, pred_mean[2,:], ribbon=2pred_std[2,:], 
          label="Predicted Predator", color=:pink, alpha=0.3)
    
    # Add title and adjust layout
    title!("Lotka-Volterra Dynamics: True vs Predicted")
    plot!(legend=:topright, grid=true)
    
    return plt
end

# Run the sampling
samples, stats = run_hmc_sampling(_p)

# Plot results
plt = plot_results(samples, ode_data)
display(plt)

savefig(plt, "lotka_volterra_results.pdf")    # PDF format (good for publications)
