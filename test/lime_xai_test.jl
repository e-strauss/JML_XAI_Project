df = CSV.read("../data/MNIST_input_9.csv", DataFrame, types=Float32)
x = Matrix(df)
y = 9

input = reshape(x, 28, 28, 1, :);
input_rgb = repeat(input, 1, 1, 3, 1)

img = load("../data/n01443537_goldfish.JPEG")
img = permutedims(channelview(img),(3,2,1))
img = reshape(img, size(img)..., 1)
#input = Float32.(img[1:32,1:32,:,1:1])
input = Float32.(img)
@info size(input)
#model = BSON.load("../data/model.bson", @__MODULE__)[:model]
model = ResNet(18; pretrain = true);
model = model.layers;
analyzer = LIME(model)
expl = analyze(input, analyzer);
heat = heatmap(expl.val)

save("../data/lime-goldfish-test.jpg", heat)
@testset "LIME XAI TEST: same model output" begin
    @test expl.output == model(input)
end