
"""
    normalize_data(X::Matrix, y::Vector, weights::Vector)

Returns the weight normalization of features and labels using a weight vector.

# Parameters
- `X::Matrix{<:Real}`: features
- `y::Vector{<:Real}`: labels
- `weights::Vector{<:Real}`: weight vector

# Returns
- Normalization of features and labels using a weight vector.
"""
function weighted_data(X::Matrix{<:Real}, y::Vector{<:Real}, weights::Vector{<:Real}) 
    X_norm =(X.-(sum(X .* weights, dims=1)./ sum(weights))) .* sqrt.(weights)
    Y_norm = (y .-(sum(y.*weights)./sum(weights))) .* sqrt.(weights)
    return X_norm, Y_norm
end

"""
    feature_selection(X::Matrix, y::Vector, max_feat::Int) -> Vector

Selects features for the model using LARS with Lasso [https://tibshirani.su.domains/ftp/lars.pdf] s.t. len(selected_features) <= max_feat
Uses the LARS package: https://github.com/simonster/LARS.jl, which needs to be installed manually via:
[package manager] add https://github.com/e-strauss/LARS.jl


# Parameters
- `X`: weighted features
- `y`: weighted labels
- `max_feat`: maximum number of feature that can be selected

# Returns
- indices of selected features
"""
function feature_selection(X::Matrix{FT}, y::Vector{FT}, max_feat::IT) where {FT<:AbstractFloat, IT<:Integer}
    c = lars(X, y; method=:lasso, intercept=false, standardize=true, lambda2=0.0,use_gram=false, maxiter=100, lambda_min=0.0, verbose=false)
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
function train_ridge_regressor(X::Matrix{<:Real}, y::Vector{<:Real}; lam::RT=1, sample_weights=I) where RT <: Real
    if sample_weights isa Vector
		W = Diagonal(sample_weights)
	else
		W = sample_weights
	end

    return inv(X'*W*X + lam*I)*X'*W*y
end

"""
    function explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, label, num_features, kernel_fn = (x) -> 1 .- x)

Takes perturbed data, labels and distances, returns explanation. Generates a relevance score for each feature based on the local approximition 
using the neighborhood_data. A relevance score close to zero means that the feature is not relevant. A larger, positive value means that the feature
contributes positively to the specific label, and negative value means that the feature decreases the chance for the classification of the specific label.

# Parameters
- `neighborhood_data`: perturbed data
- `neighborhood_labels`: corresponding perturbed labels. should have as many columns as the number of possible labels.
- `distances`: distances to original data point.
- `kernel_fn`: (similiarity) kernel function that transforms an array of distances into an array of proximity values (floats)
- `label`: label for which we want an explanation
- `num_features`: maximum number of features in explanation
- `model_regressor`: sklearn regressor to use in explanation. Defaults to Ridge regression if None. Must have model_regressor.coef_ and 'sample_weight' as a parameter to model_regressor.fit()

# Returns
- Vector{AbstractFloat}: relevance score for each feature
"""
function explain_instance_with_data(
    neighborhood_data::Matrix{FT},
    neighborhood_labels::Matrix{FT}, 
    distances::Vector{FT}, 
    label, 
    num_features::IT, 
    kernel_fn = (x) -> 1 .- x,
    lasso=true) where {FT<:AbstractFloat, IT<:Integer}

    #calculate weights using similiarity kernel function
    if kernel_fn == agnostic_kernel
        #agnostic kernel was already applied on the distances
        weights = distances
    else
        weights = kernel_fn(distances)
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

    #train a linear model on simplified features
    simplified_model = train_ridge_regressor(X[:, selected_features], y,lam=1, sample_weights=weights)
    feature_relevance = zeros(size(neighborhood_data)[2])
    feature_relevance[selected_features] .= simplified_model
    return feature_relevance 
end