using JML_XAI_Project
using Test

@testset "test_agnostic_kernel" begin
    Z1 = [
        0 1 0;
        0 1 1
    ]
    
    Z2 = [
        0 1 1 0;
        0 1 1 1;
        1 0 0 0
    ]

    @test agnostic_kernel(Z1) ≈ [0, 1//3, 1//3]
    @test agnostic_kernel(Z2) ≈ [0, 1//8, 1//4, 1//4]
end