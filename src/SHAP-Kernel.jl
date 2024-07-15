"""
    agnostic_kernel(simpleFeatures::Matrix{Float32})

Calculates the weights used for building the ridge regression model (simplified model),
    when applying SHAP with a model-agnostic kernel.

# Parameters
- `simpleFeatures`: The simplified features used for building the simlified model.

# Returns
- `Vector{Float64}`: A vector of weights with one weight for each simplified sample.

"""
function agnostic_kernel(simpleFeatures)
    weights = zeros(1)

    for i in 1:size(simpleFeatures)[1]
        
        z = simpleFeatures[i,:]
        M = length(z)
        zAbsolute = count(!iszero, z)

        weight = (M - 1)/(binomial(big(M), zAbsolute)*zAbsolute*(M - zAbsolute))
        push!(weights, weight)
    end
    
    println("Typ von weights:")
    println(typeof(weights))
    return weights
end