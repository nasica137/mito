from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from stardist.matching import matching, matching_dataset
from stardist import fill_label_holes
from stardist.models import StarDist2D
from stardist.plot import random_label_cmap
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#import glob
import cv2
from glob import glob
from csbdeep.utils import normalize
from imageio import imread
from tqdm import tqdm



# calculate average iou_score
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    
    # Handling division by zero cases
    iou = np.sum(intersection) / (np.sum(union) + np.finfo(float).eps)
    return iou


# Modify the main function
def calculate_average_iou(ground_truth_masks, predicted_masks):
    average = []
    for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
        iou = calculate_iou(pred_mask, gt_mask)
        average.append(iou)
    
    average_iou = np.sum(average) / len(average)
    return average_iou
    
    

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
    
    
    
def plot_metrics(X, Y, Y_pred, output_directory):
    """ plot with stardist"""

    taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    stats = [matching_dataset(Y, Y_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

    stats[taus.index(0.75)]

    #Plot key metrics
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend();

    # Save the figure as a PNG file
    plt.savefig(f"{output_directory}/metrics_prediction.png")
    
    """ 2nd plot"""
    
    average_iou = calculate_average_iou(Y, Y_pred)
    # Show and save all the accumulated plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Set the average IoU scores and AP values as the title
    plt.suptitle(f"Overall Average IoU Score: {average_iou:.4f}\n"
                 f"AP50: {stats[taus.index(0.5)].accuracy:.4f}  "
                 f"AP75: {stats[taus.index(0.75)].accuracy:.4f}  "
                 f"AP95: {stats[taus.index(0.95)].accuracy:.4f}", fontsize=16, y=0.95)

    for i in range(3):
        index = randint(1, len(X) - 1)
        
        # Add the generated plots to the corresponding subplots
        axes[i, 0].imshow(X[index][:, :, 0], cmap='gray')
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(Y[index][:, :, 0], cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(Y_pred[index][:, :, 0], cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')

    plt.savefig(f"{output_directory}/comparison_prediction.png", bbox_inches='tight', pad_inches=0)
    


