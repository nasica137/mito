import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_unet
import segmentation_models as sm
import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import csv
from tensorflow.keras.models import load_model



def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


def plot_and_save_loss_iou(train_loss, val_loss, loss_plot_filename, train_iou, val_iou, iou_plot_filename):
    # Plot and save the training and validation loss
    plt.plot(train_loss, 'y', label='Training loss')
    plt.plot(val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{loss_plot_filename}')
    plt.clf()

    # Plot and save the training and validation IOU
    plt.plot(train_iou, 'y', label='Training IOU')
    plt.plot(val_iou, 'r', label='Validation IOU')
    plt.title('Training and validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig(f'{iou_plot_filename}')
    plt.clf()

loss_functions = {
    "dice_loss": sm.losses.dice_loss,
    "bce_dice_loss": sm.losses.bce_dice_loss
}

def load_config(config_file):
    with open(config_file, 'r') as json_file:
        return json.load(json_file)

def setup_test_model(strategy, config):
    model_architecture = config["model_architecture"]
    learning_rate = config["learning_rate"]
    weights_filename = config["weights_filename"].format(**config)
    output_directory = config["output_directory"].format(**config)
    
    
    loss_type = config["loss"]
    loss = loss_functions.get(loss_type)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if model_architecture == 'vanilla_unet':
        input_shape = (256, 256, 3)
        preprocess_input = lambda x: x
        with strategy.scope():
            model = build_unet(input_shape)
            model.model_name = "vanilla_unet"
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',
                                                                   sm.metrics.iou_score,
                                                                   sm.metrics.f1_score,
                                                                   sm.metrics.f2_score,
                                                                   sm.metrics.recall,
                                                                   sm.metrics.precision])
    elif model_architecture[:16] == 'pretrained_unet_':
        model_name = model_architecture[16:]
        preprocess_input = sm.get_preprocessing(model_name)
        with strategy.scope():
            model = sm.Unet(model_name, encoder_weights='imagenet')
            model.model_name = model_name
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',
                                                                   sm.metrics.iou_score,
                                                                   sm.metrics.f1_score,
                                                                   sm.metrics.f2_score,
                                                                   sm.metrics.recall,
                                                                   sm.metrics.precision])
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

    model.load_weights(f"{output_directory}/{weights_filename}")

    return model, preprocess_input, optimizer, loss, output_directory
    
    
def setup_train_model(strategy, config):
    model_architecture = config["model_architecture"]
    learning_rate = config["learning_rate"]
    output_directory = config["output_directory"].format(**config)
    
    loss_type = config["loss"]
    loss = loss_functions.get(loss_type)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    

    if model_architecture == 'vanilla_unet':
        input_shape = (256, 256, 3)
        preprocess_input = lambda x: x
        with strategy.scope():
            model = build_unet(input_shape)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',
                                                                   sm.metrics.iou_score,
                                                                   sm.metrics.f1_score,
                                                                   sm.metrics.f2_score,
                                                                   sm.metrics.recall,
                                                                   sm.metrics.precision])
    elif model_architecture[:16] == 'pretrained_unet_':
        model_name = model_architecture[16:]
        preprocess_input = sm.get_preprocessing(model_name)
        with strategy.scope():
            model = sm.Unet(model_name, encoder_weights='imagenet')
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',
                                                                   sm.metrics.iou_score,
                                                                   sm.metrics.f1_score,
                                                                   sm.metrics.f2_score,
                                                                   sm.metrics.recall,
                                                                   sm.metrics.precision])
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

    return model, preprocess_input, optimizer, loss, output_directory
    

def create_test_data_generators(strategy, config, preprocess_input):
    seed = 24
    #batch_size = config["batch_size"]
    batch_size = 128
    test_images_path = config["test_images_path"]
    test_masks_path = config["test_masks_path"]

    image_data_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=lambda x: preprocess_input(x)
    )

    mask_data_generator = ImageDataGenerator(
        preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype)
    )

    test_image_generator = image_data_generator.flow_from_directory(
        test_images_path,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        seed=seed,
        shuffle=False
    )

    test_mask_generator = mask_data_generator.flow_from_directory(
        test_masks_path,
        target_size=(256, 256),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None,
        seed=seed,
        shuffle=False
    )

    total_steps = len(test_image_generator.filenames) // batch_size

    test_data_generator = my_image_mask_generator(test_image_generator, test_mask_generator)

    return test_data_generator, total_steps
    

def create_train_data_generators(strategy, config, preprocess_input, n_gpus):
    seed = 24
    batch_size = config["batch_size"]
    train_images_path = config["train_images_path"]
    train_masks_path = config["train_masks_path"]
    val_images_path = config["val_images_path"]
    val_masks_path = config["val_masks_path"]
    img_data_gen_args = config["img_data_gen_args"]
    mask_data_gen_args = config["mask_data_gen_args"]
    mask_data_gen_args['preprocessing_function'] = lambda x: np.where(x > 0, 1, 0).astype(x.dtype)

    img_data_gen_args['rescale'] = 1./255 # Normalize to [0, 1]
    img_data_gen_args['preprocessing_function'] = lambda x: preprocess_input(x)  # apply preprocessing from pretrained model


    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)



    # Training data generator
    train_image_generator = image_data_generator.flow_from_directory(
        train_images_path,
        target_size=(256, 256),  # Specify your desired image dimensions
        batch_size=batch_size,
        class_mode=None,  # Since you have separate image and mask generators
        seed=seed
    )

    train_mask_generator = mask_data_generator.flow_from_directory(
        train_masks_path,
        target_size=(256, 256),  # Specify your desired mask dimensions
        color_mode = 'grayscale',
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )

    # Validation data generator
    valid_image_generator = image_data_generator.flow_from_directory(
        val_images_path,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )

    valid_mask_generator = mask_data_generator.flow_from_directory(
        val_masks_path,
        target_size=(256, 256),
        color_mode = 'grayscale',
        batch_size=batch_size,
        class_mode=None,
        seed=seed
    )

    train_total_steps = len(train_image_generator.filenames) // (batch_size * n_gpus)
    val_total_steps = len(valid_image_generator.filenames) // (batch_size * n_gpus)
    
    train_data_generator = my_image_mask_generator(train_image_generator, train_mask_generator)

    val_data_generator = my_image_mask_generator(valid_image_generator, valid_mask_generator)

    return train_data_generator, val_data_generator, train_total_steps, val_total_steps
    

def inverse_inceptionv3_preprocess_input(x):
    # Undo scaling by multiplying by the scaling factor
    x = x * 255.0
    
    # Undo mean subtraction by adding back the mean values
    mean = [0.5, 0.5, 0.5]  # Example mean values for each channel (adjust as needed)
    x = x + mean
    
    # Clip values to be in the [0, 255] range
    x = np.clip(x, 0, 255)
    
    return x
    
def inverse_resnet_preprocess_input(x):
    
    return x
    
def inverse_vgg16_preprocess_input(x):
    # Slightly increase pixel values to make the image brighter
    brightness_factor = 3  # Adjust this factor as needed
    x = x * brightness_factor
    
    return x
    
# Define a dictionary of inverse preprocessing functions
inverse_preprocess_functions = {
    'inceptionv3': inverse_inceptionv3_preprocess_input,
    'resnet18': inverse_resnet_preprocess_input,
    'resnet34': inverse_resnet_preprocess_input,
    'resnet50': inverse_resnet_preprocess_input,
    'resnet101': inverse_resnet_preprocess_input,
    'vgg16': inverse_vgg16_preprocess_input,
    'vanilla_unet': inverse_resnet_preprocess_input,
    # Add more models and their corresponding inverse preprocessing functions here
}



def create_plot(model, test_data_generator, results, output_directory):

    num_examples = 3
    images_plotted = 0
    selected_examples = []
    
    # Get the inverse preprocessing function based on model_architecture
    model_architecture = model.model_name
    inverse_preprocess_func = inverse_preprocess_functions.get(model_architecture)
    
    if inverse_preprocess_func is None:
        print(f"Inverse preprocessing function not defined for model architecture: {model_architecture}")
        return

    for batch in test_data_generator:
        x_batch = batch[0]
        y_batch = batch[1]
        predictions = model.predict(x_batch)

        for i in range(len(x_batch)):
            original_image = x_batch[i+1]
            ground_truth = y_batch[i+1]
            prediction = predictions[i+1]
            selected_examples.append((inverse_preprocess_func(original_image), ground_truth, prediction))
            images_plotted += 1

            if images_plotted >= num_examples:
                break

        if images_plotted >= num_examples:
            break

    fig, axes = plt.subplots(num_examples, 3, figsize=(10, num_examples * 4))

    for i, (original, ground_truth, prediction) in enumerate(selected_examples):
        axes[i, 0].imshow(original, cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 1].imshow(ground_truth, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(prediction, cmap='gray')
        axes[i, 2].set_title("Model Prediction")
        
    loss, accuracy, iou, f1_score, f2_score, recall, precision = results
    
    # Specify the file path where you want to save the CSV file
    csv_file_path = f"{output_directory}/evaluation_metrics.csv"

    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the header (optional)
        writer.writerow(['Loss', 'Accuracy', 'IOU', 'F1 Score', 'F2 Score', 'Recall', 'Precision'])

        # Write the results to the CSV file
        writer.writerow(results)
    
    # Adjust the distance between subplots and title
    fig.subplots_adjust(top=0.95)

    
    fig.suptitle(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, IoU Score: {iou:.4f}, Dice Score: {f1_score:.4f}")
    plt.savefig(f"{output_directory}/comparison_prediction.png", bbox_inches='tight', pad_inches=1)
    
    
    
def evaluate_model(model, test_data_generator, total_steps):

    # Custom Progress Tracker Callback
    class ProgressTrackerCallback(tf.keras.callbacks.Callback):
        def __init__(self, total_steps):
            super(ProgressTrackerCallback, self).__init__()
            self.total_steps = total_steps

        def on_epoch_begin(self, epoch, logs=None):
            self.current_step = 0
            self.start_time = time.time()

        def on_batch_end(self, batch, logs=None):
            self.current_step += 1
            elapsed_time = time.time() - self.start_time
            time_per_batch = elapsed_time / self.current_step
            remaining_batches = self.total_steps - self.current_step
            remaining_time = time_per_batch * remaining_batches

            print(f"Processed {self.current_step}/{self.total_steps} batches. "
                  f"Estimated time remaining: {remaining_time:.2f} seconds")

    # Create the custom progress tracker
    progress_tracker = ProgressTrackerCallback(total_steps)

    # Evaluate the model using the custom progress tracker
    results = model.evaluate(test_data_generator, steps=total_steps, callbacks=[progress_tracker])

    return results



def train_model(model, train_data_generator, val_data_generator, train_total_steps, val_total_steps, config, initial_epoch=1):

    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    output_directory = config["output_directory"].format(**config)
    
    # Define a ModelCheckpoint callback to save the model weights
    checkpoint_callback = ModelCheckpoint(
        filepath= output_directory + "/weights-{epoch:02d}-{val_iou_score:.2f}.hdf5",
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        period=10  # Save every 100 epochs
    )

    # Define an EarlyStopping callback to stop training if validation loss stops improving
    early_stopping_callback = EarlyStopping(
        monitor='val_iou_score',
        mode='max',
        patience=20,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,
        restore_best_weights=True
    )


    #Fit the model
    history = model.fit(
        train_data_generator,
        validation_data=val_data_generator,
        steps_per_epoch=train_total_steps,
        validation_steps=val_total_steps,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback, early_stopping_callback],
        initial_epoch=initial_epoch)
        
    return history, model

