from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential, load_model
from src import plot_history as ph

def train(train_images, test_images, train_labels, test_labels, input_res, num_classes):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu',
                    input_shape=(input_res[0], input_res[1], 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    ## Save weights and use tensorboard
    callbacks_list = []
    filepath = "sportsballs_3c_unreg.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list.append(checkpoint)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks_list.append(tensorboard_callback)
    ## Train/Validate Model
    history = model.fit(train_images, train_labels, batch_size=64, epochs=150,
                        validation_data=(test_images, test_labels), callbacks=callbacks_list)
    # plot training session
    ph.plot_acc_loss(history, "Regularized Model")
    return model

def load_weights():
    # load weights into new model
    loaded_model = load_model("sportsballs_3c_unreg.hdf5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
    return loaded_model

## Testing
# try:
#     model = train(train_images, test_images, train_labels, test_labels, input_res, num_classes)
# except NameError:
#     train_images, test_images, train_labels, test_labels, input_res, num_classes = pd.load_data3c()
#     model = train(train_images, test_images, train_labels, test_labels, input_res, num_classes, label_dict)
