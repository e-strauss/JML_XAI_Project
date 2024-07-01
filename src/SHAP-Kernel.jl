using LinearAlgebra

function agnostic_kernel(simpleFeatures)
    M = length(simpleFeatures)
    zAbsolute = count(!iszero, simpleFeatures)

    return (M - 1)/(binomial(M, zAbsolute)*zAbsolute*(M - zAbsolute))
end