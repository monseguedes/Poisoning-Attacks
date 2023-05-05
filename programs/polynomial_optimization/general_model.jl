"""
Code to build the continuous version of 
poisoning attacks.
"""

using DynamicPolynomials
using SumOfSquares
using Ipopt
using Gurobi
using Random
using LinearAlgebra

### NLP model
# Set regularisation parameter
lambda = 0.1
model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
# Set a random seed
Random.seed!(123)
y = rand(Float64, (2,1))
y_data = [0.78, 0.34, 0.56]
y = reshape(y_data, (3,1))

num_training_data = [[0.52, 0.23] [0.43, 0.76]]
num_training = reshape(num_training_data, (2,2))

cat_training_data = [[1, 0, 0, 1] [0, 1, 1, 0]]
cat_training = reshape(cat_training_data, (2,4))
cat_poison_data = [0,1,1,0]
cat_poison = reshape(cat_poison_data, (1,4))

@variable(model, 0 <= num_poison[1:2] <= 1)
@variable(model, num_weights[1:2])
@variable(model, cat_weights[1:4])
# @variable(model, cat_poison[1:4])
@variable(model, bias)
@variable(model, t)
@objective(model, Min, -((dot(num_weights, num_training[1, :]) + dot(cat_weights, cat_training[1, :]) + bias - y[1])^2 
                        + (dot(num_weights, num_training[2, :]) + dot(cat_weights, cat_training[2, :]) + bias - y[2])^2))


for j = 1:2
    @NLconstraint(model, 2/3 * ((sum(num_weights[i] * num_training[1, i] for i in 1:2) + sum(cat_weights[i] * cat_training[1, i] for i in 1:4) + bias - y[1]) * num_training[1,j] 
                            + (sum(num_weights[i] * num_training[2, i] for i in 1:2) + sum(cat_weights[i] * cat_training[2,i] for i in 1:4) + bias - y[2]) * num_training[2, j] 
                            + t * num_poison[j] + 2 * lambda * num_weights[j]) == 0)
end

for j = 1:4
    @NLconstraint(model, 2/3 * ((sum(num_weights[i] * num_training[1,i] for i in 1:2) + sum(cat_weights[i] * cat_training[1, i] for i in 1:4) + bias - y[1]) * cat_training[1,j] 
                            + (sum(num_weights[i] * num_training[2,i] for i in 1:2) + sum(cat_weights[i] * cat_training[2, i] for i in 1:4) + bias - y[2]) * cat_training[2, j] 
                            + t * cat_poison[j] + 2 * lambda * cat_weights[j]) == 0)
end

@NLconstraint(model, 2/3 * ((sum(num_weights[i] * num_training[1,i] for i in 1:2) + sum(cat_weights[i] * cat_training[1, i] for i in 1:4) + bias - y[1]) 
                          + (sum(num_weights[i] * num_training[2,i] for i in 1:2) + sum(cat_weights[i] * cat_training[2, i] for i in 1:4) + bias - y[2]) 
                          + t)  == 0)

@NLconstraint(model, t - (sum(num_weights[i] * num_poison[i] for i in 1:2) + sum(cat_weights[i] * cat_poison[i] for i in 1:4) + bias - y[3]) == 0)  

optimize!(model)
solution_summary(model)
@show objective_value(model)


using SumOfSquares 
@polyvar num_weights[1:2] cat_weights[1:4] num_poison[1:2] bias t cat_poison[1:4]
p = -((sum(num_weights[i] * num_training[1, i] for i in 1:2) + sum(cat_weights[i] * cat_training[1, i] for i in 1:4) + bias - y[1]) 
    + (sum(num_weights[i] * num_training[2, i] for i in 1:2) + sum(cat_weights[i] * cat_training[2, i] for i in 1:4) + bias - y[2]))

S = @set num_poison[1]*(1-num_poison[1]) >= 0 && 
         num_poison[2]*(1-num_poison[2]) >= 0 

for i in 1:4    # Binary constraints
    global S = S ∩ @set(cat_poison[i] * (cat_poison[i] - 1) == 0)
end

for i in 1:2    # SOS1 constraint
    global S = S ∩ @set(sum(cat_poison[i] for i in 1:2) == 1)
end

for i in 1:2    # Num weights constraints
    global S = S ∩ @set(2/3 * ((sum(num_weights[i] * num_training[1,i] for i in 1:2) + sum(cat_weights[i] * cat_training[1,i] for i in 1:4) + bias - y[1]) * num_training[1,i] 
                            +  (sum(num_weights[i] * num_training[2,i] for i in 1:2) + sum(cat_weights[i] * cat_training[2,i] for i in 1:4) + bias - y[2]) * num_training[2,i]
                            +  (t * num_poison[i])) + (2 * lambda * num_weights[i]) == 0)
end

for i in 1:4    # Cat weights constraints
    global S = S ∩ @set(2/3 * ((sum(num_weights[i] * num_training[1,i] for i in 1:2) + sum(cat_weights[i] * cat_training[1,i] for i in 1:4) + bias - y[1]) * cat_training[1,i] 
                            +  (sum(num_weights[i] * num_training[2,i] for i in 1:2) + sum(cat_weights[i] * cat_training[2,i] for i in 1:4) + bias - y[2]) * cat_training[2,i] 
                            +  (t * cat_poison[i])) + (2 * lambda * cat_weights[i]) == 0)
end

# Bias constraint
S = S ∩ @set(2/3 * ((sum(num_weights[i] * num_training[1,i] for i in 1:2) + sum(cat_weights[i] * cat_training[1,i] for i in 1:4) + bias - y[1]) 
                 +  (sum(num_weights[i] * num_training[2,i] for i in 1:2) + sum(cat_weights[i] * cat_training[2,i] for i in 1:4) + bias - y[2]) 
                 +   t)  == 0)

println(S)

import CSDP
solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)

model = SOSModel(solver)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, p >= α, domain = S, maxdegree=2)

@constraint(model, cat_poison[2] * (cat_poison[2] - 1) == 0)

optimize!(model)
@show termination_status(model)
@show objective_value(model)
