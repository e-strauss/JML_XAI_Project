

"""
    explain_instance(image, classifier_fn, output_selection, num_features=16, num_samples=64, batch_size=5, distance_metric="cosine",)

Generates explanations for a prediction.
1. we generate neighborhood data by randomly perturbing features from the instance.
2. We then learn locally weighted linear models on this neighborhood data to explain each of the classes in an interpretable way.

# Parameters
- `image`:  4 dimension RGB image. 
- `classifier_fn`: classifier prediction probability function, which takes a image Matrix and outputs prediction probabilities.           
- `labels`: iterable with labels to be explained.          
- `num_features`: maximum number of features present in explanation, default = 16.
- `num_samples`: size of the neighborhood (perturbed images) to learn the linear model, default = 64.
- `batch_size`: batch size for model predictions, default = 8.
- `distance_metric`: the distance metric to use for weights, default = 'cosine'

# Returns:
- An ImageExplanation object with the corresponding explanations.
"""
function explain_instance(image, classifier_fn, output_selection; num_features=16, num_samples=64, batch_size=8, distance_metric="cosine", kernel=exponential_kernel, lasso=true)
    if size(image)[3] == 1
        image = reshape(image, size(image)[1:2]...)
    else
        image = permutedims(image, (3, 1, 2))
        image = RGB.(colorview(RGB, image))
    end
    # get segmentation function
    segmentation_fn = default_segmentation_function("felzenszwalb")

    # get segmentation label map
    seg_labels_map = segmentation_fn(image)
    # Make a copy of the image
    fudged_image = create_fudged_image(image, seg_labels_map)

    # more info in felzenszwalb_demo.jl

    data, labels = data_labels(image, fudged_image, seg_labels_map, classifier_fn, num_samples, batch_size)

   
    #convert integer [0 or 1] to floats
    data = Float32.(data)
    labels = Float32.(transpose(labels))

    if kernel == exponential_kernel
        distances = pairwise_distance(data, data[1:1,:], distance_metric)
    else
        distances = Float32.(kernel(data[2:end,:]))
    end

    segments_relevance_weights = explain_instance_with_data(data, labels, distances, output_selection, num_features, kernel, lasso)
    max_i, max_j = size(seg_labels_map)[1:2]
    pixel_relevance = zeros(max_i, max_j)
    for i in 1:max_i
        for j in 1:max_j
            pixel_relevance[i,j] = segments_relevance_weights[seg_labels_map[i,j]]
        end
    end
    return reshape(pixel_relevance, max_i, max_j,1,1)
end



"""
    create_fudged_image(img::Matrix{RGB{Float32}}, seg_map::Matrix{<:Integer})

Creates a "fudged" image where the pixels of each segment are replaced by the mean color of that segment.

# Parameters
- `img::Matrix{RGB{Float32}}`: The input image as a matrix of RGB colors with floating point values.
- `seg_map::Matrix{<:Integer}`: The segmentation map containing the segment labels for each pixel.

# Returns
- `Matrix{RGB{Float32}}`: A new image where the pixels of each segment are replaced by the mean color of that segment.

"""
function create_fudged_image(img::Matrix{RGB{Float32}}, seg_map::Matrix{<:Integer})
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

"""
    create_fudged_image(img::Matrix{Float32}, seg_map::Matrix{<:Integer})

Creates a "fudged" image where the pixels of each segment are replaced by the mean color of that segment.

# Parameters
- `img::Matrix{Float32}`: The input image as a matrix of RGB colors with floating point values.
- `seg_map::Matrix{<:Integer}`: The segmentation map containing the segment labels for each pixel.

# Returns
- `Matrix{RGB{Float32}}`: A new image where the pixels of each segment are replaced by the mean color of that segment.

"""
function create_fudged_image(img::Matrix{Float32}, seg_map::Matrix{<:Integer})
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
    function default_segmentation_function(algo_type::String)

Return image segmantation function, if no function was passed originally. Based on Scikit-Image implementation.

felzenszwalb:
- further explainations: https://www.analyticsvidhya.com/blog/2021/05/image-segmentation-with-felzenszwalbs-algorithm/

# Parameters
- `algo_type`: string, segmentation algorithm among the following:
        "felzenszwalb"
# Returns
- segmentation_func::Function: A segmantation function.
"""
function default_segmentation_function(algo_type::String)

    if algo_type == "felzenszwalb"
        function segmentation_func(img)
            return labels_map(felzenszwalb(img, 3, 10))
        end

    else
        error("Not a valid segmentation function!")
    end
    return segmentation_func
end


"""
    function data_labels(image::Matrix{RGB{<:AbstractFloat}}, fudged_image::Matrix{RGB{<:AbstractFloat}}, segments::Matrix{<:Integer}, classifier_fn::Function, num_samples<:Integer, batch_size<:Integer=10)

Generates perturbed versions of a given image by turning superpixels on or off,using a specified 
segmentation map. It then predicts the class probabilities for these perturbed images using a provided 
classifier function. The function returns a tuple containing the binary matrix of perturbed images (data) 
and their corresponding prediction probabilities (labels). This is useful for techniques like LIME to 
understand and explain model predictions locally.

# Parameters
- `image::Matrix{RGB{Float32}}`: The original input image.
- `fudged_image::Matrix{RGB{Float32}}`: The fudged image with mean colors for segments.
- `segments::Matrix{Int}`: The segmentation map where each segment is labeled with an integer.
- `classifier_fn::Function`: A function that takes a batch of images and returns their predicted labels.
- `num_samples::Int`:  The number of samples to generate.
- `batch_size::Int=10`: The size of the batches to process through the classifier function. Defaults to 10.

# Returns
- `data::Matrix{Int}`: A binary matrix where each row indicates which features (segments) are active.
- `labels::Matrix{Float64}`: A matrix of classifier predictions for the corresponding images in `data`.

"""
function data_labels(image::Matrix{RGB{FT}},
    fudged_image::Matrix{RGB{FT}},
    segments::Matrix{IT},
    classifier_fn,
    num_samples::IT,
    batch_size::IT=10) where {FT<:AbstractFloat, IT<:Integer}

    n_features = length(unique(segments))

    # binary matrix describing if feature/sector is replaced or not
    data = reshape(rand(0:1, n_features*num_samples), num_samples, n_features)
    
    labels = nothing

    data[1 ,:] .= 1

    imgs = nothing

    for  row in eachrow(data)

        tmp =  copy(image)

        #find all indexes where a 0 occours (this indexes will later correspond to specific features)
        indexes_where_zero = findall(x -> x == 0, row)

        # n_features x num_samples BitMatrix of type all 0 (false)
        mask = falses(size(image)...)

        # go over all segments that are supposed to be replaced and add them all together
        # (pixels of same segments should have same value in the segments map, ranging from 1 to total_number_of_segments)
        # represent each pixel to be replaced by a 1
        for index in indexes_where_zero
            mask .= mask .| (segments .== index)
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

    # add predictions to labels and empty imgs if not already done
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
    function euclidian_distance(A,B)

calculates the euclidian distance between each column vector in input matrix A and the column vectors in input matrix B

# Parameters:
- `A`:  matrix (m,n)
- `B`:  matrix (m,n) or (1,n)

# Returns:
- `distance`: 1-d array of distances
"""
function euclidian_distance(A::Matrix{FT},B::Matrix{FT}) where FT <: AbstractFloat
    difference = A .- B
    power_two = difference .^ 2
    out = sum(power_two, dims=2) .^ Float32(0.5)
    return out
end

"""
    cosine_similarity(A::AbstractArray, B::AbstractArray)

Computes the cosine similarity between corresponding rows of two arrays.

# Parameters
- `A::AbstractArray`: The first input array. Each row represents a vector.
- `B::AbstractArray`: The second input array. Each row represents a vector. Must have the same number of columns as `A`.

# Returns
- `AbstractArray`: An array of cosine similarities between corresponding rows of `A` and `B`.
"""
function cosine_similiarity(A::Matrix{FT},B::Matrix{FT}) where FT <: AbstractFloat
    scalar_product = A*B'
    norm_A = sum(A.^2, dims=2).^Float32(0.5)
    norm_B = sum(B.^2).^Float32(0.5)
    return scalar_product ./ norm_A ./ norm_B
end

"""
    cosine_similarity(A::AbstractArray, B::AbstractArray)

Computes the cosine distance: 1 - cosine_similarity(A,B)
"""
cosine_distance(A::Matrix{FT},B::Matrix{FT}) where FT <: AbstractFloat = Float32.(1.0) .- cosine_similiarity(A,B)

"""
    pairwise_distance(A::AbstractArray, B::AbstractArray; method="cosine")

Computes the pairwise distance between corresponding rows of two arrays using the specified distance metric.

# Parameters
- `A::AbstractArray`: The first input array. Each row represents a vector.
- `B::AbstractArray`: The second input array. Each row represents a vector. Must have the same number of columns as `A`.
- `method::String="cosine"`: The distance metric to use. Options are `"cosine"` for cosine similarity or `"euclidean"` for Euclidean distance. Defaults to `"cosine"`.

# Returns
- `AbstractArray`: An array of distances between corresponding rows of `A` and `B`.
"""
function pairwise_distance(A::Matrix{FT},B::Matrix{FT}, method::String="cosine") where FT <: AbstractFloat
    distance_metric = cosine_distance
    if method == "euclidian"
        distance_metric = euclidian_distance
    end
    reshape(distance_metric(A,B),:)
end

"""
    exponential_kernel(d::Vector{AbstractFloat}; kernel_width=0.25)

Computes the exponential kernel for a given array of distances.

# Parameters
- `d::Vector{AbstractFloat}`: An array of distances.
- `kernel_width::AbstractFloat=0.25`: The width of the kernel. Defaults to 0.25.

# Returns
- `Vector{AbstractFloat}`: An array of kernel values computed from the input distances.
"""
function exponential_kernel(d::Vector{FT}, kernel_width::FT=Float32(0.25)) where {FT <: AbstractFloat}
    return (exp.(.-(d.^2)) ./ kernel_width^2).^0.5
end

