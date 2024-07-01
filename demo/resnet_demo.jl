using Metalhead: ResNet
using Images

img = load("data/n01443537_goldfish.JPEG")
img = permutedims(channelview(img),(2,3,1))
img = reshape(img, size(img)..., 1)

model = ResNet(18; pretrain = true)
model = model.layers

pred = model(img)
argmax(pred)