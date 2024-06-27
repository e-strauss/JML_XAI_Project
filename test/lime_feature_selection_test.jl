using CSV
using DataFrames
using JML_XAI_Project

df = CSV.read("../data/lars_test_X.csv", DataFrame, types=Float32)
X = Matrix(df)
df = CSV.read("../data/lars_test_Y.csv", DataFrame, types=Float32)
y = vec(Matrix(df))

selection = feature_selection(X,y, 2)