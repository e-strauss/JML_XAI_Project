using JML_XAI_Project
using Test

@testset "test_agnostic_kernel" begin
    z1 = [0, 0, 1, 0, 1]
    z2 = [1, 2, 0, 0, 5, 0, 3.4]

    @test agnostic_kernel(z1) ≈ 1//15
    @test agnostic_kernel(z2) ≈ 1//70
end