using ImageSegmentation: felzenszwalb
using Images, ImageView
using Statistics
# https://juliaimages.org/v0.21/imagesegmentation


#load and display image
image = load("data/horse.jpg")

display(image)



# segments() returns instance of type SegmentedImage, contains:
#   a list of applied segment labels segment_labels(segments)
#   array containing the assigned label for each pixel (): labels_map(segments)
#   mean color and number of pixels in each segment: segment_mean(segments)

#docs: https://juliaimages.org/v0.21/function_reference/#ImageSegmentation.felzenszwalb
seg_labels_map = labels_map(felzenszwalb(image, 10, 100))

#colored_segments_image = create_image(labels_map(segments), nothing)
# image showing the segments should be opened when executing this file
#   IF all code after this is put in block comment). Idk how to open it normally


# actual usage in code:
fudged_image = copy(image)

print(length(unique(seg_labels_map)))

# iterate over segments
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

display(fudged_image)

