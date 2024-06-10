"""
    sample_data(x_0::Vector, model)

Returns the neighborhood_data (perturbed data, first element is the original data point) by sampling around x_0 
and evaluate the model to generate the corresponding perturbed labels.
"""

function sample_data(x_0, model)
    #TODO
end

"""
    normalize_data(X::Matrix, y::Vector, weights::Vector)

Returns the weight normalization of X and y using the weight vector y.
X_norm = ((X - np.average(X, axis=0, weights=weights)) * np.sqrt(weights[:, np.newaxis]))
Y_norm = ((y - np.average(y, weights=weights)) * np.sqrt(weights))
"""

function weighted_data(X, y, weights)
    #TODO
end

"""
    feature_selection(X::Matrix, y::Vector, max_feat::Int) -> ReturnType

Selects features for the model using LARS with Lasso s.t. len(selected_features) <= max_feat

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
    #TODO
end

"""
train_ridge_regressor(X::Matrix, y::Vector, weights::Vector)

Returns the trained simplified linear model using ridge regression:
model = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
model.fit(X,y, sample_weight=weights)
return model
"""

function train_ridge_regressor(X, y, weights, model)
    #TODO
end

"""Takes perturbed data, labels and distances, returns explanation.

# Parameters
- `neighborhood_data`: perturbed data, 2d array. 
- `neighborhood_labels`: corresponding perturbed labels. should have as many columns as the number of possible labels.
- `distances`: distances to original data point.
- `label`: label for which we want an explanation
- `num_features`: maximum number of features in explanation
- `model_regressor`: sklearn regressor to use in explanation. Defaults to Ridge regression if None. Must have model_regressor.coef_ and 'sample_weight' as a parameter to model_regressor.fit()

"""

function explain_instance_with_data(self,
    neighborhood_data,
    neighborhood_labels,
    distances,
    label,
    num_features,
    model_regressor=None)
    
    weights = self.kernel_fn(distances)
    X = neighborhood_data
    y = neighborhood_labels[:, label]
    X_norm, y_norm = weighted_data(X, y, weights)
    selected_features = feature_selection(X_norm, y_norm, num_features)
    simplified_model = train_ridge_regressor(X[selected_features], y, weights, model_regressor)
    #TODO: use weights of the simplified linear model for the explanation: 
    #       - high, positive weight -> positive attribution
    #       - high, negative weight -> negative attribution
    #       - low, positive or negative OR features, that were not selected by feature selection -> low attribution
end

export explain_instance_with_data