using ShapML
using DataFrames
using Random
using Statistics
using Distributed


data = DataFrame(y = 1:10, x1 = 1:10, x2 = 1:10, x3 = 1:10)

print(data)

function model_mean(data)
    y_pred = Array{Float64}(undef, size(data, 1))
    for i in 1:size(data, 1)
        y_pred[i] = mean(data[i, 2:end])
    end
    return DataFrame(y_pred = y_pred)
end

function predict_function(model, data)
    data_pred = model(data)
    return data_pred
end

data_shap = ShapML.shap(explain = data[:, 2:end],
                        model = model_mean,
                        predict_function = predict_function,
                        sample_size = 10
                        )

