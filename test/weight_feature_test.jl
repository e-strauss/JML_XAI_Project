using JML_XAI_Project

#X = [1 2 3; 4 5 6; 7 8 9]
#weights = [0.2, 0.5, 0.3]

#y =[1,2,3]

X = [1 2 3; 4 5 6; 7 8 9; 10 11 12]
weights = [0.2, 0.5, 0.3, 0.8]

y =[1,2,3,4]

X_norm, Y_norm = weighted_data(X, y, weights)

@testset "LIME XAI TEST: weighting X_norm" begin
    @test X_norm ≈ [-2.60874597 -2.60874597 -2.60874597; -2.00346921 -2.00346921 -2.00346921;  0.09128709  0.09128709  0.09128709;  2.83235277  2.83235277  2.83235277]
   # @test X_norm ≈ [-1.47580487 -1.47580487 -1.47580487; -0.21213203 -0.21213203 -0.21213203; 1.47885091  1.47885091  1.47885091]
end

@testset "LIME XAI TEST: weighting Y_norm" begin
    @test Y_norm ≈ [-0.86958199, -0.66782307,  0.03042903,  0.94411759]
    #@test Y_norm ≈ [-0.49193496, -0.07071068,  0.4929503]
end