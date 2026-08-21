from __future__ import annotations
from IPython.display import clear_output
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from neural_network import ActivationFunc, NeuralNetwork

if __name__ == "__main__":


    # ============================================================
    # 1. REPRODUCIBILITY
    # ============================================================

    np.random.seed(42)


    # ============================================================
    # 2. CREATE DATA
    # ============================================================
    
    X = np.linspace(-4 * np.pi, 4 * np.pi, 800)

    # Target function
    y = X**2 * np.sin(X)


    # ============================================================
    # 3. SCALE INPUT
    # ============================================================

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, 1)).flatten()

    # ============================================================
    # 4. CREATE NEURAL NETWORK
    # ============================================================
    nn = NeuralNetwork(inputs_count=1)
    
    # ------------------------------------------------------------
    # Hidden layer 1
    # ------------------------------------------------------------
    nn.add_layer(
        neurons_count=35,
        activation_fun= ActivationFunc.Relu,
    )

    # ------------------------------------------------------------
    # Hidden layer 2
    # ------------------------------------------------------------
    nn.add_layer(
        neurons_count=35,
        activation_fun=ActivationFunc.Relu,
    )

    # ------------------------------------------------------------
    # Output layer
    # ------------------------------------------------------------
    nn.add_layer(
        neurons_count=1
    )

    # ============================================================
    # 5. TRAINING PARAMETERS
    # ============================================================

    epochs = 800
    learning_rate = 0.001
    update_every = 1

    epoch_losses = []


    # ============================================================
    # 6. CREATE INTERACTIVE FIGURE
    # ============================================================

    plt.ion()


    fig, axes = plt.subplots(2,1,figsize=(8, 7))


    # ============================================================
    # GRAPH 1 — FUNCTION APPROXIMATION
    # ============================================================

    axes[0].plot(X, y,label="Target function",linewidth=2)

    # Початкова апроксимація
    prediction_line, = axes[0].plot(X, np.zeros_like(y),label="Neural network",linewidth=2)

    axes[0].set_xlabel("x")

    axes[0].set_ylabel("y")

    axes[0].set_title(f"Function Approximation — "f"Epoch 0/{epochs}")

    axes[0].legend()

    axes[0].grid(True)

    # ============================================================
    # GRAPH 2 — LOSS
    # ============================================================

    loss_line, = axes[1].plot([],[],linewidth=2)

    axes[1].set_xlabel("Epoch")

    axes[1].set_ylabel("Loss")

    axes[1].set_title("Training Loss")

    axes[1].grid(True)

    axes[1].set_xlim(1,epochs)


    # ============================================================
    # LAYOUT
    # ============================================================

    plt.tight_layout()


    # ============================================================
    # 7. TRAINING
    # ============================================================

    for epoch in range(epochs):
        # --------------------------------------------------------
        # Shuffle training data
        # --------------------------------------------------------

        indices = np.random.permutation(len(X_scaled))
        epoch_loss = 0.0

        # --------------------------------------------------------
        # Train each sample
        # --------------------------------------------------------

        for i in indices:
            # ----------------------------------------------------
            # Input
            # ----------------------------------------------------
            x = np.array([X_scaled[i]])

            # ----------------------------------------------------
            # Target
            # ----------------------------------------------------
            target = np.array([y[i]])

            # ----------------------------------------------------
            # Forward
            # ----------------------------------------------------
            prediction = nn.forward(x)

            # ----------------------------------------------------
            # Loss
            # ----------------------------------------------------
            sample_loss = (
                np.sum((prediction - target) ** 2)* 0.5)
            epoch_loss += sample_loss

            # ----------------------------------------------------
            # Backward
            # ----------------------------------------------------
            nn.backward(target,learning_rate)


        # --------------------------------------------------------
        # Mean epoch loss
        # --------------------------------------------------------

        epoch_loss /= len(X_scaled)

        epoch_losses.append(epoch_loss)


        # ========================================================
        # UPDATE GRAPH
        # ========================================================

        if (epoch % update_every == 0 or epoch == epochs - 1):
            # ----------------------------------------------------
            # Current predictions
            # ---------------------------------------------------- 
            predictions = np.array([nn.forward(np.array([x]))[0] for x in X_scaled])

            # ----------------------------------------------------
            # Update approximation
            # ----------------------------------------------------
            prediction_line.set_ydata(predictions)


            # ----------------------------------------------------
            # Update loss
            # ---------------------------------------------------- 
            loss_line.set_data(
                range(1,len(epoch_losses) + 1),epoch_losses)

            # ----------------------------------------------------
            # Update loss Y axis
            # ----------------------------------------------------
            max_loss = max(epoch_losses)


            if max_loss > 0:
                axes[1].set_ylim(0,max_loss * 1.1)

            # ----------------------------------------------------
            # Update titles
            # ----------------------------------------------------
            axes[0].set_title(f"Function Approximation — "f"Epoch {epoch + 1}/{epochs}")
            axes[1].set_title(f"Training Loss — "f"{epoch_loss:.8f}")

            # ----------------------------------------------------
            # Redraw existing window
            # ----------------------------------------------------
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # ----------------------------------------------------
            # Give GUI time to update
            # ----------------------------------------------------

            plt.pause(0.001)


    # ============================================================
    # 8. TURN OFF INTERACTIVE MODE
    # ============================================================
    plt.ioff()


    # ============================================================
    # 9. FINAL PREDICTIONS
    # ============================================================
    predictions = np.array([nn.forward(np.array([x]))[0] for x in X_scaled])

    # ============================================================
    # 10. METRICS
    # ============================================================

    r2 = r2_score(y,predictions)


    rmse = root_mean_squared_error(y,predictions)

    mae = mean_absolute_error(y, predictions)

    print()

    print( "================================")

    print(
        "FINAL RESULTS")

    print("================================" )

    print(f"R²   = {r2:.6f}")

    print(f"RMSE = {rmse:.6f}")


    print(f"MAE  = {mae:.6f}")