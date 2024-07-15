# Base Functions

### XAIBase.jl interface for calling LIME and SHAP
```@docs
JML_XAI_Project.LIME
JML_XAI_Project.LIME(::Any, ::JML_XAI_Project.AbstractOutputSelector)
```

### Generation of a new interpretable data set
Conversion of the input image into mulitple disturbed interpretable representations

```@docs
JML_XAI_Project.explain_instance 
JML_XAI_Project.default_segmentation_function
JML_XAI_Project.cosine_similiarity 
JML_XAI_Project.pairwise_distance 
JML_XAI_Project.data_labels  
JML_XAI_Project.euclidian_distance 
JML_XAI_Project.exponential_kernel 
JML_XAI_Project.create_fudged_image 
```

### Feature selection and train simplified model on the interpretable data set
```@docs
JML_XAI_Project.explain_instance_with_data 
JML_XAI_Project.weighted_data 
JML_XAI_Project.train_ridge_regressor
JML_XAI_Project.feature_selection
```

### Shap
```@docs
JML_XAI_Project.agnostic_kernel
```


# Index
```@index
```




