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

y = rand(Float64, (2,1))
t = rand(Float64, (1,3))

@variable(model, 0 <= x[1:3] <= 1)
@variable(model, w[1:3])
@variable(model, bias)

println(size(w))
println(size(x))
@objective(model, Max, (dot(w, t') + bias - y[1])^2)

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