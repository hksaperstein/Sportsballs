model = Sequential()
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu',
                input_shape=(input_res[0], input_res[1], 3), kernel_constraint=max_norm(4)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.5))
model.add(Dense(3))

## Save weights
model.summary()
callbacks_list = []
filepath = "sportsballs_3c_reg.hdf5"
# filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list.append(checkpoint)

## Train/Validate Model
history = model.fit(train_images, train_labels, batch_size=32, epochs=150,
                    validation_data=(test_images, test_labels), callbacks=callbacks_list)

# plot training session
ph.plot_acc_loss(history, "Initial Model")