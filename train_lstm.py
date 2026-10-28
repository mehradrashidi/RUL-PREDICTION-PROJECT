from pathlib import Path
from src.models.lstm_model import build_lstm

def train_lstm(X_train,y_train,X_val,y_val,config,output_dir="models"):
    import tensorflow as tf
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    model=build_lstm(X_train.shape[1:],units=config.get("units",64),dropout=config.get("dropout",0.2),
        recurrent_dropout=config.get("recurrent_dropout",0.0),learning_rate=config.get("learning_rate",0.001))
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=config.get("patience",10),
        restore_best_weights=True,verbose=1),tf.keras.callbacks.ModelCheckpoint(str(out/"lstm_best.keras"),
        monitor="val_loss",save_best_only=True,verbose=0),tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",factor=0.5,patience=max(3,config.get("patience",10)//2),min_lr=1e-6,verbose=1)]
    history=model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=config.get("epochs",80),
        batch_size=config.get("batch_size",128),callbacks=callbacks,verbose=2)
    return model,history
