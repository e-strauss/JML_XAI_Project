
"""
    normalize_data(X::Matrix, y::Vector, weights::Vector)

Returns the weight normalization of features and labels using a weight vector.

# Parameters
- `X`: features
- `y`: labels
-`weights`: weight vector

# Returns
"""
function weighted_data(X, y, weights)
    X_norm =(X.-(sum(X .* weights, dims=1)./ sum(weights))) .* sqrt.(weights)
    Y_norm = (y .-(sum(y.*weights)./sum(weights))) .* sqrt.(weights)
    return X_norm, Y_norm
end

"""
    feature_selection(X::Matrix, y::Vector, max_feat::Int) -> ReturnType

Selects features for the model using LARS with Lasso [https://tibshirani.su.domains/ftp/lars.pdf] s.t. len(selected_features) <= max_feat
Use LARS package: https://github.com/simonster/LARS.jl


# Parameters
- `X`: weighted features
- `y`: weighted labels

# Returns
- indices of selected features
"""
function feature_selection(X, y, max_feat)
    
    c = lars(X, y; method=:lasso, intercept=false, standardize=true, lambda2=0.0,use_gram=false, maxiter=500, lambda_min=0.0, verbose=false)
    i = size(c.coefs)[2]
    nnz_indices = findall(!iszero, c.coefs[:, i])
    while length(nnz_indices) > max_feat && i > 1
        i = i - 1
        nnz_indices = findall(!iszero, c.coefs[:, i])
    end
    return nnz_indices
end

"""
    train_ridge_regressor(X::Matrix{<:Real}, y::Vector{<:Real}; lam::Real, weights::Vector{Real})

Returns the trained simplified linear model as a matrix using ridge regression:

# Parameters
- `X::Matrix{<:Real}`: Simplified features
- `y::Vector{<:Real}`: Corresponding labels

# Returns
- `Vector{Float64}`: Simplified linear model
"""
function train_ridge_regressor(X::Matrix{<:Real}, y::Vector{<:Real}; lam::<:Real=1, sample_weights=I)
    if sample_weights isa Vector
		W = Diagonal(sample_weights)
	else
		W = sample_weights
	end

    return inv(X'*W*X + lam*I)*X'*W*y
end

"""
    function explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, label, num_features, kernel_fn = (x) -> 1 .- x)

Takes perturbed data, labels and distances, returns explanation.

# Parameters
- `neighborhood_data`: perturbed data
- `neighborhood_labels`: corresponding perturbed labels. should have as many columns as the number of possible labels.
- `distances`: distances to original data point.
- `kernel_fn`: (similiarity) kernel function that transforms an array of distances into an array of proximity values (floats)
- `label`: label for which we want an explanation
- `num_features`: maximum number of features in explanation
- `model_regressor`: sklearn regressor to use in explanation. Defaults to Ridge regression if None. Must have model_regressor.coef_ and 'sample_weight' as a parameter to model_regressor.fit()

# Returns
"""
function explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, label, num_features; kernel_fn = (x) -> 1 .- x, lasso=true)
    #calculate weights using similiarity kernel function

    if kernel_fn == exponential_kernel
        weights = kernel_fn(distances)
    else
        weights = distances
    end

    X = neighborhood_data
    #@info size(X)
    #selcted the label we want to calculate the explanation
    y = neighborhood_labels[:, label]
    #reference: python_reference/lime-base-reference.py:116
    X_norm, y_norm = weighted_data(X, y, weights)

    #select a subset of the features if 
    if lasso
        selected_features = feature_selection(X_norm, y_norm, num_features)
    else
        selected_features = collect(1:size(X_norm)[2])
    end
    
    @info "number of segments:" size(neighborhood_data)[2]
    @info "number of selected features:" length(selected_features)
    #train a linear model on simplified features
    simplified_model = train_ridge_regressor(X[:, selected_features], y,lam=1, sample_weights=weights)
    feature_relevance = zeros(size(neighborhood_data)[2])
    feature_relevance[selected_features] .= simplified_model
    return feature_relevance 
end