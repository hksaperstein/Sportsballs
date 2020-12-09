from keras.constraints import max_norm
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.models import Sequential, load_model
from src import plot_history as ph
from src import prep_data as pd
from src import gpu_mem_fix

def train(train_images, test_images, train_labels, test_labels, input_res, num_classes):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu',
                    input_shape=(input_res[0], input_res[1], 3), kernel_constraint=max_norm(4)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', kernel_constraint=max_norm(3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_constraint=max_norm(3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu', kernel_constraint=max_norm(3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    ## Save weights and use tensorboard
    callbacks_list = []
    filepath = "sportsballs_3c_reg.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list.append(checkpoint)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks_list.append(tensorboard_callback)
    ## Train/Validate Model
    history = model.fit(train_images, train_labels, batch_size=64, epochs=50,
                        validation_data=(test_images, test_labels), callbacks=callbacks_list)
    # plot training session
    ph.plot_acc_loss(history, "Regularized 3-Class Model")
    return model, history

def load_weights():
    # load weights into new model
    loaded_model = load_model("sportsballs_3c_reg.hdf5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return loaded_model

## Testing
# try:
#     model, history = train(train_images3, test_images3, train_labels3, test_labels3, input_res, 3)
# except NameError:
#     train_images3, test_images3, train_labels3, test_labels3, input_res, num_classes, label_dict = pd.load_data3c()
#     model, history = train(train_images3, test_images3, train_labels3, test_labels3, input_res, 3)
