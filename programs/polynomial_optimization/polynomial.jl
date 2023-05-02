"""
Code to build the polynomial optimization version of 
poisoning attacks.
"""

using DynamicPolynomials
using SumOfSquares
using Ipopt
using Random
using LinearAlgebra

## Online example 
@polyvar x y
p = x^3 - x^2 + 2x*y -y^2 + y^3
using SumOfSquares
S = @set x >= 0 && y >= 0 && x + y >= 1 
p(x=>1, y=>0), p(x=>1//2, y=>1//2), p(x=>0, y=>1)

# Ipopt 
using Ipopt
model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
@variable(model, a >= 0)
@variable(model, b >= 0)
@constraint(model, a + b >= 1)
peval(a, b) = p(x=>a, y=>b)
register(model, :peval, 2, peval, autodiff=true)
@NLobjective(model, Min, peval(a, b))
optimize!(model)
@show termination_status(model)
@show value(a)
@show value(b)
@show objective_value(model)

# Polynomial 
import CSDP
solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, p >= α, domain = S)
optimize!(model)
@show termination_status(model)
@show objective_value(model)