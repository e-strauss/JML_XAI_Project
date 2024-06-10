"""Selects features for the model. see explain_instance_with_data to
understand the parameters."""

function feature_selection()
    #TODO
end

"""Takes perturbed data, labels and distances, returns explanation.

Args:
neighborhood_data: perturbed data, 2d array. first element is
assumed to be the original data point.
neighborhood_labels: corresponding perturbed labels. should have as
many columns as the number of possible labels.
distances: distances to original data point.
label: label for which we want an explanation
num_features: maximum number of features in explanation
feature_selection: how to select num_features. options are:
'forward_selection': iteratively add features to the model.
This is costly when num_features is high
'highest_weights': selects the features that have the highest
product of absolute weight * original data point when
learning with all the features
'lasso_path': chooses features based on the lasso
regularization path
'none': uses all features, ignores num_features
'auto': uses forward_selection if num_features <= 6, and
'highest_weights' otherwise.
model_regressor: sklearn regressor to use in explanation.
Defaults to Ridge regression if None. Must have
model_regressor.coef_ and 'sample_weight' as a parameter
to model_regressor.fit()

Returns:
(intercept, exp, score, local_pred):
intercept is a float.
exp is a sorted list of tuples, where each tuple (x,y) corresponds
to the feature id (x) and the local weight (y). The list is sorted
by decreasing absolute value of y.
score is the R^2 value of the returned explanation
local_pred is the prediction of the explanation model on the original instance
"""

function explain_instance_with_data(self,
    neighborhood_data,
    neighborhood_labels,
    distances,
    label,
    num_features,
    feature_selection="auto",
    model_regressor=None)
    #TODO
end


