def train_model(model, train_data_generator, val_data_generator, train_total_steps, val_total_steps, config, initial_epoch=1):
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    output_directory = config["output_directory"].format(**config)

    # Create an empty dictionary to store the training history
    training_history = {'loss': [], 'accuracy': [], 'iou_score': [], 'f1_score': [], 'f2_score': [],
                        'recall': [], 'precision': [], 'val_loss': [], 'val_accuracy': [],
                        'val_iou_score': [], 'val_f1_score': [], 'val_f2_score': [],
                        'val_recall': [], 'val_precision': []}

    # Define a ModelCheckpoint callback to save the model weights and training history
    class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
        def on_epoch_end(self, epoch, logs=None):
            # Save the training history
            for key, value in logs.items():
                if key in training_history:
                    training_history[key].append(value)
            # Call the parent on_epoch_end method to save the model weights
            super().on_epoch_end(epoch, logs)

    # Define a CustomModelCheckpoint callback to save both model weights and training history
    checkpoint_callback = CustomModelCheckpoint(
        filepath=output_directory + "/weights-{epoch:02d}-{val_iou_score:.2f}.hdf5",
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        period=10  # Save every 100 epochs
    )

    # Define an EarlyStopping callback to stop training if validation loss stops improving
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_iou_score',
        mode='max',
        patience=20,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,
        restore_best_weights=True
    )

    # Fit the model
    history = model.fit(
        train_data_generator,
        validation_data=val_data_generator,
        steps_per_epoch=train_total_steps,
        validation_steps=val_total_steps,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping_callback],
        initial_epoch=initial_epoch
    )

    # Save the model weights at the end of training
    model.save_weights(f"{output_directory}/final_weights.hdf5")

    # Save the training history to a CSV file
    history_csv_path = f"{output_directory}/training_history.csv"
    pd.DataFrame(training_history).to_csv(history_csv_path, index=False)

    return history, model
