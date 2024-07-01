using ImageSegmentation: felzenszwalb
using Images: labels_map,colorview,channelview, RGB, Gray, red, green, blue, N0f8
using Statistics: mean
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
function explain_instance(image, classifier_fn, output_selection, num_features=8, num_samples=64, batch_size=5, distance_metric="cosine",)
    if size(image)[3] == 1
        image = reshape(image, size(image)[1:2]...)
    else
        @info size(image)
        image = permutedims(image, (3, 1, 2))
        image = RGB.(colorview(RGB, image))
    end
    # get segmentation function
    segmentation_fn = default_segmentation_function("felzenszwalb")

    # get segmentation label map
    seg_labels_map = segmentation_fn(image)
    @info "nums segs:" length(unique(seg_labels_map))
    # Make a copy of the image
    fudged_image = create_fudged_image(image, seg_labels_map)

    # more info in felzenszwalb_demo.jl

    data, labels = data_labels(image, fudged_image, seg_labels_map, classifier_fn, num_samples, batch_size)
    data = Float32.(data)
    labels = transpose(labels)
    distances = pairwise_distance(data, data[1:1,:], distance_metric)

    segments_relevance_weights = explain_instance_with_data(data, labels, distances, output_selection, num_features, exponential_kernel)
    max_i, max_j = size(seg_labels_map)[1:2]
    pixel_relevance = zeros(max_i, max_j)
    for i in 1:max_i
        for j in 1:max_j
            pixel_relevance[i,j] = segments_relevance_weights[seg_labels_map[i,j]]
        end
    end
    #TODO: build relevance weights for all pixel of the original image using segments_relevance_weights and segemnts
    return reshape(pixel_relevance, max_i, max_j,1,1)
end

function create_fudged_image(img::Matrix{RGB{Float32}}, seg_map)
    fudged_image = copy(img)
    for segment_label in unique(seg_map)
        mask = (seg_map .== segment_label)
    
        mean_color = RGB(
            mean([red(c) for c in img[mask]]),
            mean([green(c) for c in img[mask]]),
            mean([blue(c) for c in img[mask]])
        )
        
        # Apply the mean color to all pixels in the current segment
        fudged_image[mask] .= mean_color
    end
    return fudged_image
end

function create_fudged_image(img::Matrix{Float32}, seg_map)
    fudged_image = copy(img)
    for segment_label in unique(seg_map)
        mask = (seg_map .== segment_label)
        mean_color = mean(img[mask])
        
        # Apply the mean color to all pixels in the current segment
        fudged_image[mask] .= mean_color
    end
    return fudged_image
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
            return labels_map(felzenszwalb(img, 10, 10))
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
    
    labels = nothing

    # make first row all 1s / all features enabled
    data[1 ,:] .= 1

    imgs = nothing

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
        if typeof(tmp) === Matrix{RGB{Float32}}
            
            tmp = permutedims(channelview(tmp),(3,2,1))
            tmp = reshape(tmp,size(tmp)...,1)
        else
            tmp = reshape(tmp,size(tmp)...,1,1)
        end

        
        if imgs === nothing
            imgs = tmp
        else
            imgs = cat(imgs,tmp,dims=4)
        end

        # if batch size is reached: add predictions to labels and empty imgs
        if size(imgs)[4] == batch_size
            preds = classifier_fn(imgs)
            if labels === nothing
                labels = preds
            else
                labels = cat(labels,preds,dims=2)
            end
            imgs = nothing
        end
    end

    # add predictions to labels and empty imgs if not alreadydone
    if (imgs !== nothing) && (size(imgs)[4] > 0)
        preds = classifier_fn(imgs)
        if labels === nothing
            labels = preds
        else
            labels = cat(labels,preds,dims=2)
        end
    end

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
function euclidian_distance(A,B)
    difference = A .- B
    power_two = difference .^ 2
    return sum(power_two, dims=2) .^ 0.5
end

function cosine_similiarity(A, B) 
    scalar_product = A*B'
    norm_A = sum(A.^2, dims=2).^0.5
    norm_B = sum(B.^2).^0.5
    return scalar_product ./ norm_A ./ norm_B
end

cosine_distance(A,B) = 1.0 .- cosine_similiarity(A,B)

function pairwise_distance(A, B, method="cosine")
    distance_metric = cosine_distance
    if method == "euclidian"
        distance_metric = euclidian_distance
    end
    reshape(distance_metric(A,B),:)
end

#if kernel is None:
#    def kernel(d, kernel_width):
#        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
function exponential_kernel(d, kernel_width=0.25)
    return (exp.(.-(d.^2)) ./ kernel_width^2).^0.5
end

