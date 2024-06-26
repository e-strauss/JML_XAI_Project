using JML_XAI_Project
using Test

@testset "test_train_ridge_regressor" begin
    X = [
        1 2;
        2 4;
        3 6
    ]

    y = [2, 4, 8]
    w = [9, 5, 7]
    lambda = 3.5

    @test train_ridge_regressor(X, y) ≈ [57//71 -28//71; -28//71 15//71]*[34, 68]
    @test train_ridge_regressor(X,y, lamb=lambda, sample_weights=w) ≈ [-0.03425 0.02321; 0.02321 -0.01205]*[226, 452]
end