using ImageSegmentation: felzenszwalb
using Images

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

    top = labels

    data, labels = data_labels(image, fudged_image, seg_labels_map, classifier_fn, num_samples, batch_size)
    

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


"""
Generates perturbed versions of a given image by turning superpixels on or off,using a specified 
segmentation map. It then predicts the class probabilities for these perturbed images using a provided 
classifier function. The function returns a tuple containing the binary matrix of perturbed images (data) 
and their corresponding prediction probabilities (labels). This is useful for techniques like LIME to 
understand and explain model predictions locally.
"""
function data_labels(image, fudged_image, segments, classifier_fn, num_samples, batch_size=10)

    #number of features/segments in segmented image
    n_features = length(unique(segments))

    # binary matrix consisting of (row) vectors describing if feature is replaced or not
    data = reshape(rand(0:1, n_features*num_samples), num_samples, n_features)
    
    labels = []

    # make first row all 1s / all features enabled
    data[1 ,:] .= 1

    imgs = []

    for  row in eachrow(data)

        tmp =  copy(image)

        #find all indexes where a 0 occours (this indexes will later correspond to specific features)
        zeros_indexes = findall(x -> x == 0, row)

        # n_features x num_samples BitMatrix of type all 0 (false)
        mask = falses(size(segments)...)

        # go over all segments that are supposed to be replaced and add them all together
        # (pixels of same segments should have same value in the segments map, ranging from 1 to total_number_of_segments)
        # represent each pixel to be replaced by a 1
        for zero_index in zeros_indexes
            mask .= mask .| (segments .== zero_index)
        end

        # replace marked parts in copy of original image
        tmp[mask] = fudged_image[mask]

        push!(imgs, tmp)

        # if batch size is reached: add predictions to labels and empty imgs
        if length(imgs) == batch_size
            preds = classifier_fn(imgs)
            push!(labels, preds)
            imgs = Array[]
        end
    end

    # add predictions to labels and empty imgs if not alreadydone
    if length(imgs) > 0
        preds = classifier_fn(imgs)
        push!(labels, preds)
    end

    print(typeof(data), typeof(labels))
    return data, labels
end


"""
calculates the euclidian distance between each column vector in input matrix A and the column vectors in input matrix B

Args:
    A:  matrix (m,n)
    B:  matrix (m,n) or (1,n)
Returns:
    distance: 1-d array of distances
"""
function pairwise_distance(A, B)
    difference = A .- B
    power_two = difference .^ 2
    distances = sum(power_two, dims=2) .^ 0.5
    return reshape(distances, :)
end