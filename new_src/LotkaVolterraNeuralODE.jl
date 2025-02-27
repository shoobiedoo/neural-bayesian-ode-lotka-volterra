# Previous imports remain the same
using SciMLSensitivity, DifferentialEquations
using Lux, Zygote
using Random, Plots, AdvancedHMC, MCMCChains, StatsPlots, ComponentArrays
using JLD2, CSV, DataFrames
using Statistics

# Minimal configuration struct with only the requested parameters
struct MinimalConfig
    step_size::Float64
    n_samples::Int
    n_adapts::Int
    lambda_l1::Float64
    lambda_l2::Float64
    description::String  # Kept for saving results
end

# Default configuration
function default_minimal_config()
    MinimalConfig(
        0.25,   # step_size
        500,    # n_samples
        500,    # n_adapts
        0.0,    # lambda_l1
        1.0,    # lambda_l2
        "lotka_volterra_simulation"
    )
end

# Fixed parameters
const target_acceptance_rate = 0.45
const lv_params = (α=1.5, β=1.0, γ=3.0, δ=1.0)
const tspan = (0.0, 10.0)
const u0 = [1.0; 1.0]
const datasize = 40

# Generate training data with fixed parameters
function generate_training_data()
    tsteps = range(tspan[1], tspan[2], length=datasize)
    
    function lotka_volterra(du, u, p, t)
        du[1] = -lv_params.α*u[1] - lv_params.β*u[1]*u[2]
        du[2] = -lv_params.δ*u[2] + lv_params.γ*u[1]*u[2]
    end
    
    prob = ODEProblem(lotka_volterra, u0, tspan)
    Array(solve(prob, Tsit5(), saveat=tsteps))
end

# Modified HMC setup with minimal configurable parameters
function setup_hmc(p, config::MinimalConfig, ode_data, predict_neural_ode)
    metric = DiagEuclideanMetric(length(p))
    
    function l(θ)
        pred = predict_neural_ode(θ)
        mse_loss = -sum(abs2, ode_data .- pred)
        l1_reg = -config.lambda_l1 * sum(abs, θ)
        l2_reg = -config.lambda_l2 * sum(abs2, θ)
        return mse_loss + l1_reg + l2_reg
    end
    
    function dldθ(θ)
        x, lambda = Zygote.pullback(l, θ)
        grad = first(lambda(1))
        return x, grad
    end
    
    h = Hamiltonian(metric, l, dldθ)
    integrator = Leapfrog(config.step_size)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), 
                            StepSizeAdaptor(target_acceptance_rate, integrator))
    
    return h, kernel, adaptor
end

# Run HMC sampling with minimal config
function run_hmc_sampling(p, config::MinimalConfig, ode_data, predict_neural_ode)
    h, kernel, adaptor = setup_hmc(p, config, ode_data, predict_neural_ode)
    samples, stats = sample(h, kernel, p, config.n_samples, adaptor, config.n_adapts; 
                          progress=true)
    return samples, stats
end

# Save results
function save_simulation_results(samples, ode_data, config::MinimalConfig, predict_neural_ode)
    tsteps = range(tspan[1], tspan[2], length=datasize)
    predictions = cat([predict_neural_ode(samples[i]) for i in 1:length(samples)]..., dims=3)
    
    pred_mean = mean(predictions, dims=3)[:,:,1]
    pred_std = std(predictions, dims=3)[:,:,1]
    
    # Save results with minimal configuration
    simulation_state = Dict(
        "config" => Dict(
            "step_size" => config.step_size,
            "n_samples" => config.n_samples,
            "n_adapts" => config.n_adapts,
            "lambda_l1" => config.lambda_l1,
            "lambda_l2" => config.lambda_l2
        ),
        "predictions" => Dict(
            "mean" => pred_mean,
            "std" => pred_std
        ),
        "true_data" => ode_data,
        "samples" => samples
    )
    
    @save "$(config.description)_results.jld2" simulation_state
    
    return pred_mean, pred_std, tsteps
end

# Convert data to appropriate types to avoid method call warnings
function create_plot(tsteps, ode_data, pred_mean, pred_std)
    # Convert inputs to vectors to ensure type stability
    t = collect(tsteps)
    true_prey = vec(ode_data[1,:])
    true_predator = vec(ode_data[2,:])
    pred_prey_mean = vec(pred_mean[1,:])
    pred_prey_std = vec(pred_std[1,:])
    pred_predator_mean = vec(pred_mean[2,:])
    pred_predator_std = vec(pred_std[2,:])

    # Create new figure
    p = plot(dpi=300)  # Create empty plot with higher resolution
    
    # Add each line separately
    plot!(p, t, true_prey, 
          label="True Prey",
          color=:blue,
          linewidth=2)
    
    plot!(p, t, true_predator,
          label="True Predator",
          color=:red,
          linewidth=2)
    
    # Add predictions with uncertainty bands
    plot!(p, t, pred_prey_mean,
          ribbon=2 .* pred_prey_std,
          fillalpha=0.3,
          label="Predicted Prey",
          color=:lightblue)
    
    plot!(p, t, pred_predator_mean,
          ribbon=2 .* pred_predator_std,
          fillalpha=0.3,
          label="Predicted Predator",
          color=:pink)
    
    # Customize the plot
    plot!(p, 
          xlabel="Time",
          ylabel="Population",
          title="Lotka-Volterra Dynamics: True vs Predicted",
          legend=:topright,
          grid=true,
          framestyle=:box)
    
    return p
end
# Function to save plots in multiple formats
function save_plots(p, basename)
    savefig(p, basename * ".pdf")
    savefig(p, basename * ".png")
end

# Main execution function
function run_simulation(config::MinimalConfig=default_minimal_config())
    # Generate training data
    ode_data = generate_training_data()
    tsteps = range(tspan[1], tspan[2], length=datasize)
    
    # Setup neural network (fixed architecture)
    dudt2 = Lux.Chain(x -> x .^ 3,
                      Lux.Dense(2, 50, tanh),
                      Lux.Dense(50, 2))
    
    Random.seed!(123)
    rng = Random.default_rng()
    p, st = Lux.setup(rng, dudt2)
    _p = ComponentArray{Float64}(p)
    
    # Neural ODE functions
    function neural_ode_func(u, p, t)
        dudt2(u, p, st)[1]
    end
    
    function prob_neural_ode(u0, p)
        prob = ODEProblem(neural_ode_func, u0, tspan, p)
        solve(prob, Tsit5(), saveat=tsteps)
    end
    
    function predict_neural_ode(p)
        Array(prob_neural_ode(u0, p))
    end
    
    # Run HMC sampling
    samples, stats = run_hmc_sampling(_p, config, ode_data, predict_neural_ode)
    
    # Save and plot results
    pred_mean, pred_std, tsteps = save_simulation_results(samples, ode_data, config, predict_neural_ode)
    
    plt = create_plot(tsteps, ode_data, pred_mean, pred_std)
    
    save_plots(plt, config.description)
    
    return plt, samples, stats, ode_data
end


custom_config = MinimalConfig(
    0.65,  # step_size
    500,   # n_samples
    500,    # n_adapts
    0.0,   # lambda_l1
    0.0,    # lambda_l2
    "base_500_step_0.65"
)

# Run with custom parameters
plt_custom, samples_custom, stats_custom, ode_data_custom = run_simulation(custom_config)