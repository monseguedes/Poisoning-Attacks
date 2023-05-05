
using DynamicPolynomials
using SumOfSquares

@polyvar num_poison[1:3] x[1:3]
#p = sum(x)^2
S = algebraicset([xi^2 - 1 for xi in x])


S = @set num_poison[2]*(1-num_poison[2]) >= 0
for i in 1:3
    global S = S ∩ @set(num_poison[i]*(1-num_poison[i]) >= 0)
end

println(S)
