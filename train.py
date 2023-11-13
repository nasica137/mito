from __future__ import print_function, unicode_literals, absolute_import, division
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from other_utils import load_config, setup_train_model, create_train_data_generators, train_model, create_plot, plot_and_save_loss_iou
import os
import pandas as pd    

strategy = tf.distribute.MirroredStrategy()
n_gpus = strategy.num_replicas_in_sync
print('Number of devices: %d' % n_gpus)


# Load configuration from JSON
config = load_config('config3.json')
model_architecture = config["model_architecture"]
loss_plot_filename = config["loss_plot_filename"].format(**config)
iou_plot_filename = config["iou_plot_filename"].format(**config)
weights_filename = config["weights_filename"].format(**config)

# Unpack configuration variables
model, preprocess_input, optimizer, loss, output_directory = setup_train_model(strategy, config)

# Create the directory for this configuration
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Data Generators
train_data_generator, val_data_generator, train_total_steps, val_total_steps = create_train_data_generators(strategy, config, preprocess_input, n_gpus)


# Evaluate the model using the custom progress tracker
history, model = train_model(model, train_data_generator, val_data_generator, train_total_steps, val_total_steps, config)

# Save the model weights
model.save_weights(f"{output_directory}/{weights_filename}")

# Call the plot_and_save_loss_iou function from utils to plot and save loss and IOU
plot_and_save_loss_iou(
    history.history['loss'],
    history.history['val_loss'],
    f"{output_directory}/{loss_plot_filename}",
    history.history['iou_score'],
    history.history['val_iou_score'],
    f"{output_directory}/{iou_plot_filename}")
    

print(f"Training {model_architecture} is done.")
