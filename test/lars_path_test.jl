using LARS
using CSV
using DataFrames

df = CSV.read("data/lars_test_X.csv", DataFrame)
X = Matrix(df)
df = CSV.read("data/lars_test_Y.csv", DataFrame)
y = vec(Matrix(df))

@info size(X) size(y)

c = lars(X, y; method=:lasso, intercept=false, standardize=true, lambda2=0.0,
     use_gram=false, maxiter=500, lambda_min=0.0, verbose=false)

@info c.lambdas
display(c.coefs)