import tensorflow as tf
from tensorflow import keras

class ConvergenceCallback(keras.callbacks.Callback):

    def __init__(self,initial_lr,
                 loss_threshold=1e-2,
                 consecutive_epochs=5,
                 lr_ratio=0.1):
        super().__init__()
        self.initial_lr = initial_lr
        self.loss_threshold = loss_threshold
        self.consecutive_epochs = consecutive_epochs
        self.lr_threshold = initial_lr * lr_ratio

        self.previous_val_loss = None
        self.stable_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_loss = logs.get('val_loss')
        if val_loss is None:
            return

        if self.previous_val_loss is not None:
            loss_change = abs(val_loss - self.previous_val_loss)

            if loss_change < self.loss_threshold:
                self.stable_epochs += 1
            else:
                self.stable_epochs = 0

        self.previous_val_loss = val_loss

        current_lr = float(
            tf.keras.backend.get_value(
                self.model.optimizer.learning_rate))

        loss_converged = (self.stable_epochs >= self.consecutive_epochs)
        lr_decayed = current_lr <= self.lr_threshold

        if loss_converged and lr_decayed:
            print(f"\nEmpirical convergence reached at epoch {epoch + 1}.")
            print(f"Current learning rate: {current_lr:.8g}")
            self.model.stop_training = True