from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from neural_network import ActivationFunc, LossFunc, NeuralNetwork

img = Image.open("digit_2.png")

X = np.array(img)
X = X / 255.0
X = X.flatten()

INPUT_SIZE = X.shape[0]
LATENT_SIZE = 14

autoencoder = NeuralNetwork(inputs_count=INPUT_SIZE)
autoencoder.add_layer(neurons_count=128,activation_fun= ActivationFunc.Relu)
#autoencoder.add_layer(neurons_count= 64, activation_fun=ActivationFunc.Relu)
#autoencoder.add_layer(neurons_count=LATENT_SIZE,activation_fun= ActivationFunc.Relu)
#autoencoder.add_layer(neurons_count= 64, activation_fun=ActivationFunc.Relu)
#autoencoder.add_layer(neurons_count=128,activation_fun= ActivationFunc.Relu)
#autoencoder.add_layer(neurons_count=INPUT_SIZE,activation_fun= ActivationFunc.Sigmoid)

epochs = 300
for epoch in range(epochs):
    output = autoencoder.forward(X)

    autoencoder.backward(train_data=X, learning_rate=0.001,lossFunc=LossFunc.MSE)

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss = {autoencoder.loss[-1]:.6f}"
        )

reconstructed = autoencoder.forward(X)
reconstructed_image = reconstructed.reshape(28, 28)
reconstructed_image = reconstructed_image * 255

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.show()