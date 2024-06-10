# LIME

Because LIME is a "Local Interpretable **Model-agnostic** Explanations" (LIME) you can use it for any model. In addition to the model to be explained, you need a specific input x.

For our example we use a pre-trained LeNet5 model (model.bson) and a data point of the MNIST dataset for x.

```@example implementations
using ExplainableAI
using Flux
using BSON
using JML_XAI_Project
using CSV
using DataFrames
using XAIBase
using Images

#our input x
df = CSV.read("../MNIST_input_9.csv", DataFrame)
x = Matrix(df)
y = 9

input = reshape(x, 28, 28, 1, :);
input_rgb = repeat(input, 1, 1, 3, 1)

#our model
model = BSON.load("../model.bson", @__MODULE__)[:model]

```
Now we are using the XAIBase template to generate an explanaition

```@example implementations
struct LIME{M} <: AbstractXAIMethod 
    model::M    
end

function (method::LIME)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    val = rand(size(input)...) #it doesn't implement LIME yet
    extras = nothing  # TODO: optionally add additional information using a named tuple
    return Explanation(val, output, output_selection, :MyMethod, :attribution, extras)
end
```

In the next step we call the XAIBase's analyze function to compute the LIME explanation.

```@example implementations
analyzer = LIME(model)
expl = analyze(input, analyzer);
```
To visualize the explanaition we create a heatmap
```@example implementations
using VisionHeatmaps
heatmap(expl.val)
```
!!! note

    As you can see, the heatmap is quite noisy. This is because the XAIBase tamplate is still generating a random output. We still have to implement the LIME algoithm. But if you want to have a first impression of the LIME code structure, have a look at our source code. 