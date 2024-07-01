using Images, ImageView

include("src/Lime-images.jl")

#print("cwd:", dirname(pwd()))

img = load("./data/4x4_pixel.jpg")

img_white = load("./data/4x4_pixel_all_white.jpg")

#display(img_white)
# 1: orange
# 2: pink
# 3: yellow
# 4: blue
# 5: red
# 6: green 

segments = [1 2 2 3
            1 1 2 3
            4 4 3 3
            4 4 5 6]

dumb_classifier(input) = "duck"



data, labels = data_labels(img, img_white, segments, dumb_classifier, 2)

