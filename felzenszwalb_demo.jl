using ImageSegmentation: felzenszwalb
using Images, ImageView
using MLDatasets
using Random
using Plots
using Colors
using Statistics
# https://juliaimages.org/v0.21/imagesegmentation

"""
Create an image based on the number of each entry in a matrix
by simply mapping each number to a ranomized color (same numbers have same color)
or the mean color of the individual segment.
Used for visualization in this file.
"""
function create_image(matrix, color_mapping=nothing)
    unique_numbers = unique(matrix)
    if color_mapping == nothing
        color_map = Dict(num => RGB(rand(), rand(), rand()) for num in unique_numbers)
    else
        color_map = color_mapping
    end

    height, width = size(matrix)
    img = Array{RGB{Float64}}(undef, height, width)

    for i in 1:height
        for j in 1:width
            img[i, j] = color_map[matrix[i, j]]
        end
    end

    return img
end

#load and display image
image_matrix, label = MNIST(Float32, :test)[12]
img = convert2image(MNIST, image_matrix)
#display(img)


max_segment_number = 10
min_cluster_size = 4 # smallest number of pixels/region, has big influence on result

# segments() returns instance of type SegmentedImage, contains:
#   a list of applied segment labels segment_labels(segments)
#   array containing the assigned label for each pixel (): labels_map(segments)
#   mean color and number of pixels in each segment: segment_mean(segments)

segments = felzenszwalb(img, max_segment_number, min_cluster_size)

colored_segments_img = create_image(labels_map(segments), nothing)
# image showing the segments should be opened when executing this file
#   IF all code after this is put in block comment). Idk how to open it normally


# actual usage in code:
fudged_image = copy(image_matrix)

for segment_label in unique(labels_map(segments))
    mask = labels_map(segments) .== segment_label

    mean_color = mean(image_matrix[mask, :], dims=1)
    
    # Broadcast the mean color to all pixels in the current segment
    fudged_image[mask, :] .= mean_color
end


display(Gray.(fudged_image))

print(unique(fudged_image))

 