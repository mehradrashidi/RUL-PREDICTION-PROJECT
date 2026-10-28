def build_lstm(input_shape, units=64, dropout=0.2, recurrent_dropout=0.0, learning_rate=0.001):
    try: import tensorflow as tf
    except ImportError as exc: raise ImportError("TensorFlow is required for LSTM. Install requirements.txt.") from exc
    model=tf.keras.Sequential([tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout),
        tf.keras.layers.Dense(max(units//2,16),activation="relu"),tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model
