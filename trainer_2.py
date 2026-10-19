import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from image_preprocess_2 import generate_data_2

from Setting import data
from Setting import hyperparameters
from tensorflow.keras.callbacks import ReduceLROnPlateau

from training_callbacks import ConvergenceCallback
from DLIM import build_DLIB_model


# ============================================================
# Build Model
# ============================================================
model = build_DLIB_model()

class DLIM_Trainer(tf.keras.Model):

    def __init__(self, dlim_model):

        super().__init__()

        self.dlim = dlim_model
        
        # ====================================================
        # loss
        # ====================================================
        self.ce = tf.keras.losses.CategoricalCrossentropy()
        self.bce = tf.keras.losses.CategoricalCrossentropy()
        
        # ====================================================
        # metrics
        # ====================================================
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.ld_tracker = tf.keras.metrics.Mean(name="LD")
        self.ll_tracker = tf.keras.metrics.Mean(name="LL")

        self.acc_P = tf.keras.metrics.CategoricalAccuracy(name="acc_P")
        self.acc_Q = tf.keras.metrics.CategoricalAccuracy(name="acc_Q")

    @property
    def metrics(self):

        return [self.total_loss_tracker,
                self.ld_tracker,
                self.ll_tracker,
                self.acc_P,
                self.acc_Q]

    # ========================================================
    # train step
    # ========================================================
    def train_step(self, data):
        x, (y1, y2) = data
        y2 = tf.nn.softmax(y2/3)

        with tf.GradientTape() as tape:

            P, Q = self.dlim(x, training=True)
            
            Q = tf.nn.softmax(Q/3)

            LD = self.ce(y1, P)
            LL = self.bce(y2, Q)

            total_loss = LL + 0.5 * LD

        grads = tape.gradient(total_loss,
                              self.dlim.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.dlim.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.ld_tracker.update_state(LD)
        self.ll_tracker.update_state(LL)

        # accuracy
        self.acc_P.update_state(y1, P)
        self.acc_Q.update_state(y2, Q)

        return {"loss": self.total_loss_tracker.result(),
                "LD": self.ld_tracker.result(),
                "LL": self.ll_tracker.result(),
                "acc_P": self.acc_P.result(),
                "acc_Q": self.acc_Q.result()}

    # ========================================================
    # val step
    # ========================================================
    def val_step(self, data):
        x, (y1, y2) = data
        
        P, Q = self.dlim(x, training=False)

        LD = self.ce(y1, P)
        LL = self.bce(y2, Q)

        total_loss = LL + 0.5 * LD

        self.total_loss_tracker.update_state(total_loss)
        self.ld_tracker.update_state(LD)
        self.ll_tracker.update_state(LL)

        self.acc_P.update_state(y1, P)
        self.acc_Q.update_state(y2, Q)

        return {"loss": self.total_loss_tracker.result(),
                "LD": self.ld_tracker.result(),
                "LL": self.ll_tracker.result(),
                "acc_P": self.acc_P.result(),
                "acc_Q": self.acc_Q.result()}



# ============================================================
# Create Trainer
# ============================================================

trainer = DLIM_Trainer(model)

# The following values should be generated according to own dataset.
y2_train_all = np.load("DLIM_y2_train_labels.npy").astype(np.float32)
y2_train_all = tf.constant(y2_train_all)

y2_val_all = np.load("DLIM_y2_val_labels.npy").astype(np.float32)
y2_val_all = tf.constant(y2_val_all)


train_gen = generate_data_2(data.path_1,
                            data.label,
                            y2_path=y2_train_all,
                            height=hyperparameters.height,
                            width=hyperparameters.width,
                            batch_size=hyperparameters.batch_size_t,
                            num=hyperparameters.num)

path_val  = data.path_2
label_val = data.label
val_gen = generate_data_2(path_val,
                          label_val,
                          y2_path=y2_val_all,
                          height=hyperparameters.height,
                          width=hyperparameters.width,
                          batch_size=hyperparameters.batch_size_v,
                          num=hyperparameters.num)


reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.8,
                              patience=1,
                              mode='auto',
                              verbose=1,
                              min_lr=1e-6)


initial_lr = hyperparameters.lr

convergence_callback = ConvergenceCallback(initial_lr=initial_lr,
                                           loss_threshold=1e-2,
                                           consecutive_epochs=5,
                                           lr_ratio=0.1)


optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

trainer.compile(optimizer=optimizer)

trainer.fit(train_gen,
            steps_per_epoch=hyperparameters.steps_t,
            epochs=hyperparameters.epochs,
            validation_data=val_gen,
            validation_steps=hyperparameters.steps_v,
            callbacks=[reduce_lr,
                     convergence_callback])

model.save("./")




