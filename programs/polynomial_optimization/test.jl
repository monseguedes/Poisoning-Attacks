
using DynamicPolynomials

@polyvar x[1:3] y[1:3]
#p = sum(x)^2
using SumOfSquares
S = AbstractSemialgebraicSet([xi^2 - 1 for xi in x])
