using ImageSegmentation: felzenszwalb

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

    # get segmentation function
    segmentation_fn = default_segmentation_function("felzenszwalb")

    # get segmentation label map
    seg_labels_map = segmentation_fn(image)

    # Make a copy of the image
    fudged_image = copy(image)

    # do we need this?
    if hide_color === nothing


        for segment_label in unique(seg_labels_map)
            mask = seg_labels_map .== segment_label
        
            mean_color = RGB(
                mean([red(c) for c in image[mask]]),
                mean([green(c) for c in image[mask]]),
                mean([blue(c) for c in image[mask]])
            )
            
            # Apply the mean color to all pixels in the current segment
            fudged_image[mask] .= mean_color
        end
    end
    # more info in felzenszwalb_demo.jl


    #TODO reference to python implementation ../python-reference/lime-image.py
end 

"""
return image segmantation function, if no function was passed
originally based on Scikit-Image implementation
julia adaptations:

quickshift
- package: ??? (docs: ???)
- explaination: ???
- python packages can be used in julia, it's therefore possible to use the scikit-image library if desired

slic
- code: https://github.com/Cuda-Chen/SLIC.jl/tree/master (docs: NOT EVEN AN INOFFICIAL PACKAGE)
- code seems to work, but spelling mistakes in the original and takes forever
- explaination: https://cuda-chen.github.io/image%20processing/2020/07/29/slic-in-julia.html

felzenszwalb
- package: ImageSegmentation (docs: https://juliaimages.org/v0.21/imagesegmentation/)
- explaination: https://www.analyticsvidhya.com/blog/2021/05/image-segmentation-with-felzenszwalbs-algorithm/

comparision: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html

Args:
    algo_type: string, segmentation algorithm among the following:
        'quickshift', 'slic', 'felzenszwalb'
    target_params: dict, algorithm parameters (valid model paramters
        as define in Scikit-Image documentation)


"""
function default_segmentation_function(algo_type::String)

    if algo_type== "felzenszwalb"
        function segmentation_func(img)
            return labels_map(felzenszwalb(img, 10, 100))
        end

    else
        error("Not a valid segmentation function!")
    end
    return segmentation_func
end