import LinearAlgebra.I
import LinearAlgebra.Diagonal
import LARS.lars

"""
    sample_data(x0::Vector, x0_pertubed::Vector, H, D, model)

Returns the neighborhood_data (perturbed data, first element is the original data point) by sampling around x0 
and evaluate the model to generate the corresponding perturbed labels.

Citation from Paper [https://arxiv.org/pdf/1602.04938]:
We sample instances around x0_pertubed by drawing nonzero elements of x0_pertubed uniformly at random (where the number of such draws is also uniformly sampled).
We sample instances both in the vicinity of x0 and far away from x (measured by the distance metric).


# Parameters
- `x0`: original input
- `x0_pertubed`: interpretable representation of original input
- `H`: interpretable representation mapping function, s.t. h(x0_pertubed) = x0
- `D`: distance metric which calculates the distance of two points in the origial, non-interpretable space
- `model`: model which works on the non-interpretable original input

# Returns
- `neighborhood_data`: sampled perturbed data (first element is x0_pertubed)
- `neighborhood_labels`: corresponding perturbed calculated by the model and mapping function H
- `distances`: distances of each sampled point to original data point calculated using the distance metric D and mapping function H
"""
function sample_data(x_0, model)
    #TODO
end

"""
    normalize_data(X::Matrix, y::Vector, weights::Vector)

Returns the weight normalization of X (data) and y (label) using the weight vector y.
X_norm = ((X - np.average(X, axis=0, weights=weights)) * np.sqrt(weights[:, np.newaxis]))
Y_norm = ((y - np.average(y, weights=weights)) * np.sqrt(weights))
"""
function weighted_data(X, y, weights)
    X_norm =(X.-(sum(X .* weights, dims=1)./ sum(weights))) .* sqrt.(weights)
    Y_norm = (y .-(sum(y.*weights)./sum(weights))) .* sqrt.(weights)
    return X_norm, Y_norm
end

export weighted_data

"""
    feature_selection(X::Matrix, y::Vector, max_feat::Int) -> ReturnType

Selects features for the model using LARS with Lasso [https://tibshirani.su.domains/ftp/lars.pdf] s.t. len(selected_features) <= max_feat
Use LARS package: https://github.com/simonster/LARS.jl

Python reference:
nonzero = range(weighted_data.shape[1])
coefs = generate_lars_path(X, y)
for i in range(len(coefs.T) - 1, 0, -1):
    nonzero = coefs.T[i].nonzero()[0]
    if len(nonzero) <= num_features:
        return nonzero

# Parameters
- `X`: weighted feature
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
train_ridge_regressor(X::Matrix, y::Vector, weights::Vector)

Returns the trained simplified linear model using ridge regression:
model = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
model.fit(X,y, sample_weight=weights)
return model
"""
function train_ridge_regressor(X, y; lam=1, sample_weights=I)
    if sample_weights isa Vector
		W = Diagonal(sample_weights)
	else
		W = sample_weights
	end
    return inv(X'*W*X + lam*I)*X'*W*y
end

export train_ridge_regressor

"""Takes perturbed data, labels and distances, returns explanation.

# Parameters
- `neighborhood_data`: perturbed data
- `neighborhood_labels`: corresponding perturbed labels. should have as many columns as the number of possible labels.
- `distances`: distances to original data point.
- `kernel_fn`: (similiarity) kernel function that transforms an array of distances into an array of proximity values (floats)
- `label`: label for which we want an explanation
- `num_features`: maximum number of features in explanation
- `model_regressor`: sklearn regressor to use in explanation. Defaults to Ridge regression if None. Must have model_regressor.coef_ and 'sample_weight' as a parameter to model_regressor.fit()

"""
function explain_instance_with_data(neighborhood_data, neighborhood_labels, distances, label, num_features, kernel_fn = (x) -> 1 .- x)
    #calculate weights using similiarity kernel function 
    weights = kernel_fn(distances)

    X = neighborhood_data
    #@info size(X)
    #selcted the label we want to calculate the explanation
    y = neighborhood_labels[:, label]
    #reference: python_reference/lime-base-reference.py:116
    X_norm, y_norm = weighted_data(X, y, weights)

    #select a subset of the features
    selected_features = feature_selection(X_norm, y_norm, num_features)
    @info "number of segments:" size(neighborhood_data)[2]
    @info "number of selected features:" length(selected_features)
    #train a linear model on simplified features
    simplified_model = train_ridge_regressor(X[:, selected_features], y,lam=1, sample_weights=weights)
    feature_relevance = zeros(size(neighborhood_data)[2])
    feature_relevance[selected_features] .= simplified_model
    return feature_relevance 
end

export feature_selection