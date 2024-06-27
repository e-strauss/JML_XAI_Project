using JML_XAI_Project
using Test

@testset "test_train_ridge_regressor" begin
    X = [
        1 2;
        2 4;
        3 6
    ]

    y = [2, 4, 8]
    w = [1, 3, 5]
    lambda = 2

    @test train_ridge_regressor(X, y) ≈ [57//71 -28//71; -28//71 15//71]*[34, 68]
    @test train_ridge_regressor(X,y; lam=lambda, sample_weights=w) ≈ [117//292 -58//292; -58//292 30//292]*[146, 292]
end