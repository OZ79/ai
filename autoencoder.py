from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from neural_network import ActivationFunc, LossFunc, NeuralNetwork


# -------------------------
# 1. Завантаження картинки
# -------------------------

img = Image.open("Lenna.png").convert("L")

X = np.array(img, dtype=np.float32)
X = X / 255.0
IMAGE_HEIGHT, IMAGE_WIDTH = X.shape
X = X.flatten()
INPUT_SIZE = X.shape[0]


# -------------------------
# 2. Створення шуму
# -------------------------

noise_factor = 0.3

noise = np.random.normal(
    loc=0.0,
    scale=noise_factor,
    size=X.shape
)

X_noisy = X + noise
X_noisy = np.clip(X_noisy, 0.0, 1.0)
noisy_image = X_noisy.reshape(IMAGE_WIDTH, IMAGE_HEIGHT) * 255


# -------------------------
# 3. Autoencoder
# -------------------------
LATENT_SIZE = 16

autoencoder = NeuralNetwork(inputs_count=INPUT_SIZE)
autoencoder.add_layer(neurons_count=128,activation_fun=ActivationFunc.Relu)
autoencoder.add_layer(neurons_count=64,activation_fun=ActivationFunc.Relu)
autoencoder.add_layer(neurons_count=LATENT_SIZE,activation_fun=ActivationFunc.Relu)
autoencoder.add_layer(neurons_count=64,activation_fun=ActivationFunc.Relu)
autoencoder.add_layer(neurons_count=128,activation_fun=ActivationFunc.Relu)
autoencoder.add_layer(neurons_count=INPUT_SIZE,activation_fun=ActivationFunc.Sigmoid)


# -------------------------
# 4. Навчання
# -------------------------
epochs = 200
for epoch in range(epochs):
    output = autoencoder.forward(X)
    autoencoder.backward(train_data=X,learning_rate=0.001,lossFunc=LossFunc.MSE)

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss = {autoencoder.loss[-1]:.6f}"
        )


# -------------------------
# 5. Реконструкція
# -------------------------

reconstructed = autoencoder.forward(X_noisy)
reconstructed_image = reconstructed.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
reconstructed_image = reconstructed_image * 255

# Тестуємо на noisy 2
noise_2 = np.random.normal(
    loc=0.0,
    scale=0.3,
    size=X.shape
)
X_noisy_2 = noise_2
X_noisy_2 = np.clip(X_noisy_2, 0.0, 1.0)
noisy_image_2 = X_noisy_2.reshape(IMAGE_WIDTH, IMAGE_HEIGHT) * 255

reconstructed_2 = autoencoder.forward(X_noisy_2)
reconstructed_image_2 = reconstructed_2.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
reconstructed_image_2 = reconstructed_image_2 * 255


# -------------------------
# 6. Відображення
# -------------------------

plt.figure(figsize=(14, 4))


plt.subplot(1, 5, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")


plt.subplot(1, 5, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy")
plt.axis("off")


plt.subplot(1, 5, 3)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Noisy")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(noisy_image_2, cmap="gray")
plt.title("Noisy 2")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(reconstructed_image_2, cmap="gray")
plt.title("Reconstructed Noisy 2")
plt.axis("off")

plt.show()