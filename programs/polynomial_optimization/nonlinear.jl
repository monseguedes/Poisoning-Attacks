"""
Code to build the continuous version of 
poisoning attacks.
"""

using DynamicPolynomials
using SumOfSquares
using Ipopt
using Random
using LinearAlgebra

### NLP model
# Set regularisation parameter
lambda = 0.1
model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
# Set a random seed
Random.seed!(123)
y = rand(Float64, (2,1))
t = rand(Float64, (1,3))
@variable(model, 0 <= x[1:3] <= 1)
@variable(model, w[1:3])
@variable(model, bias)
println(size(w))
println(size(x))
println(size(t'))
@objective(model, Min, -(dot(w, t') + bias - y[1])^2)


for i = 1:3
    @NLconstraint(model, 2 * ((sum(w[i] * t[i] for i in 1:3) + bias - y[1]) * t[i] 
                            + (sum(w[i] * x[i] for i in 1:3) + bias - y[2]) * x[i]) + 2 * lambda * w[i] == 0)
    # @constraint(model, 2 * ((dot(w,t') + bias - y[1]) * t[i] 
    #                       + (dot(w,t') + bias - y[2]) * x[i]) + 2 * lambda * w[i] == 0)
end
@NLconstraint(model, 2 * ((sum(w[i] * t[i] for i in 1:3) + bias - y[1]) 
                        + (sum(w[i] * x[i] for i in 1:3) + bias - y[2]))  == 0)

optimize!(model)
solution_summary(model)
@show objective_value(model)

@polyvar w1 w2 w3 x1 x2 x3 bias 
p = -((w1 * t[1] + w2 * t[2] + w3 * t[3] + bias - y[1])^2)
using SumOfSquares
S = @set x1*(1-x1) >= 0 && x2*(1-x2) >= 0 && x3*(1-x3) >= 0 &&
        2*((sum(w1*t[1]+w2*t[2]+w3*t[3])+bias-y[1])*t[1] +
        (sum(w1*x1+w2*x2+w3*x3)+bias-y[2])*x1) + 2*lambda*w1 == 0 &&
        2*((sum(w1*t[1]+w2*t[2]+w3*t[3])+bias-y[1])*t[2] +
        (sum(w1*x1+w2*x2+w3*x3)+bias-y[2])*x2) + 2*lambda*w2 == 0 &&
        2*((sum(w1*t[1]+w2*t[2]+w3*t[3])+bias-y[1])*t[3] +
        (sum(w1*x1+w2*x2+w3*x3)+bias-y[2])*x3) + 2*lambda*w3 == 0 &&
        2*((sum(w1*t[1]+w2*t[2]+w3*t[3])+bias-y[1]) +
        (sum(w1*x1+w2*x2+w3*x3)+bias-y[2])) == 0

import CSDP
solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true)

model = SOSModel(solver)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, p >= α, domain = S)
optimize!(model)
@show termination_status(model)
@show objective_value(model)
