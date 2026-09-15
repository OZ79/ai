import os

import kagglehub
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from neural_network import ActivationFunc, LossFunc, NeuralNetwork

import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("shrutisaxena/raisin-dataset")
df = pd.read_excel(os.path.join(path, 'Raisin_Dataset.xlsx'))


mtx = df.drop('Class', axis=1).corr(numeric_only=True).abs()

X = df.drop(columns="Class").copy()
y = df["Class"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# Спочатку поділ даних
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MI обчислюємо лише на навчальній вибірці
mi_scores = mutual_info_classif(
    X_train,
    y_train,
    discrete_features=False,
    random_state=42
)

mi_scores = pd.Series(
    mi_scores,
    name="MI Scores",
    index=X_train.columns
).sort_values(ascending=False)

# Відбір ознак: MI > 0
useful_features = mi_scores[
    mi_scores > 0
].index.tolist()


# Кореляцію також обчислюємо лише на X_train
corr_matrix = X_train[
    useful_features
].corr().abs()

upper = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)


to_drop = set()

for col in upper.columns:
    for row in upper.index:

        if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.95:

            if mi_scores[row] < mi_scores[col]:
                to_drop.add(row)
            else:
                to_drop.add(col)


final_features = [
    f
    for f in useful_features
    if f not in to_drop
]

# Залишаємо однакові ознаки у train і test
X_train = X_train[final_features]
X_test = X_test[final_features]


print("\nВідібрані ознаки для моделі:")
print(final_features)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#########################################

nn = NeuralNetwork(inputs_count=X_train.shape[1])
nn.add_layer(neurons_count=32,activation_fun= ActivationFunc.Relu)
nn.add_layer(neurons_count=1,activation_fun= ActivationFunc.Sigmoid)

#########################################

epochs = 160
learning_rate = 0.0001

epoch_losses = []

# Інтерактивний режим matplotlib
plt.ion()

fig, ax = plt.subplots(figsize=(10, 5))

line, = ax.plot([], [], label="Training Loss")

ax.set_title("Neural Network Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend()

for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))

    epoch_loss = []

    for i in indices:

        x = X_train[i]
        target = np.array([y_train[i]])

        # Forward
        prediction = nn.forward(x)

        # Backward
        nn.backward(target, learning_rate,  lossFunc = LossFunc.BCE)

        # Loss поточного sample
        epoch_loss.append(nn.loss[-1])

    # Середній Loss за epoch
    mean_loss = np.mean(epoch_loss)
    epoch_losses.append(mean_loss)

    # ==========================
    # ОНОВЛЕННЯ ГРАФІКА
    # ==========================

    line.set_data(
        range(1, len(epoch_losses) + 1),
        epoch_losses
    )

    # Оновлюємо межі осей
    ax.relim()
    ax.autoscale_view()

    # Перемальовуємо
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Даємо GUI час оновити вікно
    plt.pause(0.001)

plt.ioff()
plt.show()

prediction = np.array([nn.forward(x)[0] for x in X_test])

# Формування прогнозованих класів:
# ймовірність >= 0.5 — клас 1
# ймовірність < 0.5 — клас 0
y_pred = (prediction >= 0.5).astype(int)

# Обчислення частки правильних прогнозів
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nРезультати NN")
print(f"Accuracy: {accuracy:.4f}")

# Виведення матриці помилок:
# TN — правильно визначений клас 0
# FP — клас 0 помилково визначений як 1
# FN — клас 1 помилково визначений як 0
# TP — правильно визначений клас 1
print("\nМатриця помилок:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# Виведення метрик для кожного класу
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Kecimen", "Besni"],
        zero_division=0
    )
)

# ROC-AUC
fpr, tpr, thresholds = roc_curve(y_test, prediction)
auc = roc_auc_score(y_test, prediction)

plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


