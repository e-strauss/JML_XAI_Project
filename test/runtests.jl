using JML_XAI_Project
using Test

@testset "JML_XAI_Project.jl" begin
    @testset "timestwo" begin
        @test timestwo(4.0) == 8.0
    end
end
