# How does LIME works


### How does the code works for image input
1. Create super pixels for the input image through segmentation
2. Create different disturbed versions of the input image
3. Predict the class probabilities of disturbed images with classificator
4. Calculate distance of features of original and disturbed Images
5. Calculate Explanaition - calculate weights of each superpixel:
    - kernel function is applyed on the calculated distances used as weights
    - normalize features and labels using weights
    - select n features with lasso
    - calculate ridge regression model as simplified model with selected features and distance weights 
    - create a vector with the length corresponding to the number of features, in place of the selected features store their ridge regression coefficients, the other entries are 0
6. Give each segment in the image its calculated weight

