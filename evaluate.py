from __future__ import print_function, unicode_literals, absolute_import, division
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from other_utils import load_config, setup_test_model, create_test_data_generators, evaluate_model, create_plot

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)

# Load configuration from JSON
config = load_config('config3.json')

# Unpack configuration variables
model, preprocess_input, optimizer, loss, output_directory = setup_test_model(strategy, config)

# Data Generators
test_data_generator, total_steps = create_test_data_generators(strategy, config, preprocess_input)

# Evaluate the model using the custom progress tracker
results = evaluate_model(model, test_data_generator, total_steps)
print(results)

# Create and save a comparison plot
create_plot(model, test_data_generator, results, output_directory)
