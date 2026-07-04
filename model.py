"""
model.py
--------
Defines the Keras LSTM model architecture for stock price forecasting.

The architecture uses two stacked LSTM layers with dropout for regularisation,
a Dense bottleneck layer, and a single-neuron output layer.

Usage (standalone)
------------------
    python model.py          # prints a model summary
"""

import os

import config

# Suppress TensorFlow info/warning noise at import time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model as _keras_load_model


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def build_lstm_model(
    lookback: int = config.LOOKBACK_WINDOW,
    lstm_units_1: int = config.LSTM_UNITS_1,
    lstm_units_2: int = config.LSTM_UNITS_2,
    dropout_rate: float = config.DROPOUT_RATE,
    dense_units: int = config.DENSE_UNITS,
    output_units: int = config.OUTPUT_UNITS,
    learning_rate: float = config.LEARNING_RATE,
) -> keras.Model:
    """Build and compile a stacked LSTM model for time-series forecasting.

    Architecture
    ------------
    Input → LSTM(128, return_sequences=True) → Dropout
          → LSTM(64) → Dropout
          → Dense(25, relu)
          → Dense(1)

    Parameters
    ----------
    lookback : int
        Number of past time-steps fed into the model (sequence length).
    lstm_units_1 : int
        Number of units in the first LSTM layer.
    lstm_units_2 : int
        Number of units in the second LSTM layer.
    dropout_rate : float
        Fraction of units to drop for regularisation.
    dense_units : int
        Number of neurons in the intermediate Dense layer.
    output_units : int
        Number of output neurons (1 for univariate forecasting).
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    keras.Model
        Compiled model ready for training.
    """
    model = keras.Sequential(
        [
            # First LSTM block – returns full sequences for stacking
            layers.Input(shape=(lookback, 1), name="input_layer"),
            layers.LSTM(
                lstm_units_1,
                return_sequences=True,
                name="lstm_1",
            ),
            layers.Dropout(dropout_rate, name="dropout_1"),
            # Second LSTM block – returns only the last hidden state
            layers.LSTM(
                lstm_units_2,
                return_sequences=False,
                name="lstm_2",
            ),
            layers.Dropout(dropout_rate, name="dropout_2"),
            # Dense bottleneck
            layers.Dense(dense_units, activation="relu", name="dense_1"),
            # Output layer (linear activation for regression)
            layers.Dense(output_units, name="output"),
        ],
        name="StockLSTM",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: keras.Model, path: str = config.MODEL_SAVE_PATH) -> None:
    """Save a trained Keras model in the native ``.keras`` format.

    Parameters
    ----------
    model : keras.Model
    path : str
        Destination file path (should end in ``.keras``).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)
    print(f"[model] Model saved to '{path}'.")


def load_model(path: str = config.MODEL_SAVE_PATH) -> keras.Model:
    """Load a previously saved ``.keras`` model from disk.

    Parameters
    ----------
    path : str
        Path to the saved model file.

    Returns
    -------
    keras.Model

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``path``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at '{path}'. "
            "Run train.py first to create one."
        )
    model = _keras_load_model(path)
    print(f"[model] Model loaded from '{path}'.")
    return model


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    m = build_lstm_model()
    m.summary()
