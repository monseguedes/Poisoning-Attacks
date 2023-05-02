using JuMP, CSDP, SumOfSquares

# Define the number of variables and constraints
n_numerical_features = 3
n_categrocial_features = 3
n_categories = 3
n_samples = 2
n_training_samples = 1


# Define the variables and parameters
@polyvar w[1:n_numerical_features] x[1:n_numerical_features] bias
# Set a random seed
Random.seed!(123)
y = rand(Float64, (n_samples,1))
t = rand(Float64, (1,n_numerical_features))
lambda = 0.1

# Define the objective function
p = -(sum((sum(w[i] * t[i] for i in 1:n_numerical_features) + bias - y[j])^2 for j in 1:n_training_samples))

# Define the set of constraints
S = @set [x[i]*(1-x[i]) >= 0 for i in 1:n_numerical_features]... # `...` is the splat operator to unpack arrays
for j in 1:n_numerical_features
    c = 2 * ((sum(w[i] * t[i] for i in 1:n_numerical_features) + bias - y[n_training_samples]) * t[j] + 
             (sum(w[i] * x[i] for i in 1:n_numerical_features) + bias - y[n_training_samples + 1]) * x[j]) +
        2 * lambda * w[j]
    push!(S, c == 0)
end

# Define the optimizer
solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)

# Define the SOS model
model = SOSModel(solver)

# Define the variables
@variable(model, α)

# Define the objective function
@objective(model, Max, α)

# Define the constraint
@constraint(model, p >= α, domain = S)

# Optimize the model
optimize!(model)

# Print the results
@show termination_status(model)
@show objective_value(model)
