"""Generates explanations for a prediction.

First, we generate neighborhood data by randomly perturbing features
from the instance (see __data_inverse). We then learn locally weighted
linear models on this neighborhood data to explain each of the classes
in an interpretable way (see lime_base.py).

Args:
    image: 3 dimension RGB image. If this is only two dimensional,
        we will assume it's a grayscale image and call gray2rgb.
    classifier_fn: classifier prediction probability function, which
        takes a numpy array and outputs prediction probabilities.  For
        ScikitClassifiers , this is classifier.predict_proba.
    labels: iterable with labels to be explained.
    hide_color: If not None, will hide superpixels with this color.
        Otherwise, use the mean pixel color of the image.
    top_labels: if not None, ignore labels and produce explanations for
        the K labels with highest prediction probabilities, where K is
        this parameter.
    num_features: maximum number of features present in explanation
    num_samples: size of the neighborhood to learn the linear model
    batch_size: batch size for model predictions
    distance_metric: the distance metric to use for weights.
    model_regressor: sklearn regressor to use in explanation. Defaults
    to Ridge regression in LimeBase. Must have model_regressor.coef_
    and 'sample_weight' as a parameter to model_regressor.fit()
    segmentation_fn: SegmentationAlgorithm, wrapped skimage
    segmentation function
    random_seed: integer used as random seed for the segmentation
        algorithm. If None, a random integer, between 0 and 1000,
        will be generated using the internal random number generator.
    progress_bar: if True, show tqdm progress bar.

Returns:
    An ImageExplanation object (see lime_image.py) with the corresponding
    explanations.
"""

function explain_instance(self, image, classifier_fn, labels=(1,),
    hide_color=None,
    top_labels=5, num_features=100000, num_samples=1000,
    batch_size=10,
    segmentation_fn=None,
    distance_metric="cosine",
    model_regressor=None,
    random_seed=None,)
    
    #TODO reference to python implemenation ../python-reference/lime-image.py
end 
