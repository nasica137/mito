import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from patchify import patchify, unpatchify
import segmentation_models_pytorch.utils as smp_utils
import csv
import segmentation_models_pytorch as smp


# Load configuration from JSON file
with open('finetune-config1.json', 'r') as config_file:
    config = json.load(config_file)
    
output_directory = config['output_directory'].format(**config)
# Model architecture configuration
model_config = config["model"]
model_name = model_config["name"]
encoder_name = model_config["encoder_name"]

from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')


# Load the model checkpoint
model_checkpoint_path = f'./fine-tuning/{output_directory}/best_model.pth'  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_checkpoint_path, map_location=device) # needs to be loaded with 4 devices if training was done on 4 devices!

# Set the model to evaluation mode
model.eval()

resize_size = 768

# Load and preprocess the image
image_path = 'original_image.png'  
image = Image.open(image_path).resize((resize_size,resize_size)).convert('RGB')
image = np.array(image)

image_for_plot = image.copy() 

# Apply preprocessing function
image = preprocess_input(image)


# Load and preprocess the mask
mask_path = 'ground_truth_mask.png'
ground_truth_mask = Image.open(mask_path).resize((resize_size, resize_size))
ground_truth_mask = np.array(ground_truth_mask).astype(np.uint8)

#print(image.shape)
print("start patching large image...")
# Define patch size and step for overlapping tiles
patch_size = (256, 256, 3)
step = 128

# Make predictions for each patch with overlapping tiles
predicted_mask = np.zeros((resize_size, resize_size), dtype=np.float32)
count_map = np.zeros((resize_size, resize_size), dtype=np.float32)

for y in range(0, resize_size - patch_size[0] + 1, step):
    for x in range(0, resize_size - patch_size[1] + 1, step):
        # Extract patch
        patch = image[y:y + patch_size[0], x:x + patch_size[1]]
        patch = np.transpose(patch, (2, 0, 1))  # Change shape to match model input shape
        
        # Convert patch to tensor
        patch_tensor = torch.from_numpy(patch).to(device).float().unsqueeze(0)
        
        # Perform prediction on the patch
        with torch.no_grad():
            output = model(patch_tensor).sigmoid()
            output = output.cpu().numpy().squeeze().astype(np.float32)
        
        # Update predicted_mask and count_map with the patch predictions
        predicted_mask[y:y + patch_size[0], x:x + patch_size[1]] += output
        count_map[y:y + patch_size[0], x:x + patch_size[1]] += 1

# Take the average of the predictions in overlapping regions
print(np.unique(count_map))
print(np.unique(predicted_mask))
predicted_mask = predicted_mask / count_map
print(np.unique(predicted_mask))

# Threshold the predicted mask
threshold = 0.5
predicted_mask_binary = (predicted_mask > threshold).astype(np.uint8)


print(np.unique(predicted_mask))
#for i in np.unique(predicted_mask):
#    print(i)
print(ground_truth_mask)




# Convert the predicted and ground truth masks to tensors
predicted_mask_tensor = torch.from_numpy(predicted_mask)
ground_truth_mask_tensor = torch.from_numpy(ground_truth_mask)



# List of thresholds
thresholds = [0.5, 0.75, 0.85, 0.95]

# Define the CSV file name
csv_filename = f'./fine-tuning/{output_directory}/large_image_metrics.csv'

# Open the CSV file in write mode
with open(csv_filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(['Threshold', 'IoU Score', 'F1 Score', 'F2 Score', 'Accuracy', 'Recall'])

    # Compute metrics for each threshold
    for threshold in thresholds:
        tp, fp, fn, tn = smp.metrics.get_stats(predicted_mask_tensor, ground_truth_mask_tensor, mode='binary', threshold=threshold)
        
        # Compute metrics using the obtained stats for each threshold
        iou_score = round(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item(), 4)
        f1_score = round(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item(), 4)
        f2_score = round(smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro").item(), 4)
        accuracy = round(smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item(), 4)
        recall = round(smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise").item(), 4)

        # Write metrics for the current threshold to the CSV file
        csv_writer.writerow([threshold, iou_score, f1_score, f2_score, accuracy, recall])

print(f"Metrics logged in {csv_filename}")


"""
# Calculate metrics for the predicted and ground truth masks
results = {}
for metric in metrics:
    metric_name = metric.__class__.__name__
    results[metric_name] = metric(predicted_mask_tensor, ground_truth_mask_tensor)
    print(metric(predicted_mask_tensor, ground_truth_mask_tensor))

# Print or use the results as needed
for metric_name, value in results.items():
    print(f"{metric_name}: {value}")"""


















# Convert the NumPy array to a PIL image
pred_pil = Image.fromarray((predicted_mask_binary * 255).astype(np.uint8))


# Save the PIL image as a PNG file
pred_pil.save(f'./fine-tuning/{output_directory}/predicted_mask.png')


# Plotting the original image and predicted mask
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_for_plot)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Ground Truth Mask')
plt.imshow(ground_truth_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Predicted Mask')
plt.imshow(predicted_mask_binary, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig(f'./fine-tuning/{output_directory}/test_large_image.png')



import SimpleITK as sitk
import matplotlib.pyplot as plt

# Load images
moving_image1 = sitk.ReadImage(f'ground_truth_mask.png', sitk.sitkFloat32)
moving_image2 = sitk.ReadImage(f'./fine-tuning/{output_directory}/predicted_mask.png', sitk.sitkFloat32)
fixed_image = sitk.ReadImage(f'original_image.png', sitk.sitkFloat32)

# Display the images
plt.figure(figsize=(10, 5))

# Display fixed image in gray color
plt.imshow(sitk.GetArrayViewFromImage(fixed_image), cmap='gray')

# Overlay moving image1 (ground truth) in 'coolwarm' colormap with higher opacity
plt.imshow(sitk.GetArrayViewFromImage(moving_image1), cmap='coolwarm', alpha=0.7)

# Overlay moving image2 (predicted) in 'viridis' colormap with lower opacity
plt.imshow(sitk.GetArrayViewFromImage(moving_image2), cmap='viridis', alpha=0.3)

plt.axis('off')
plt.title('Fixed Image (Gray) & Ground Truth Overlay (Red - Coolwarm) & Predicted Overlay (Blue - Viridis)')
plt.savefig(f'./fine-tuning/{output_directory}/test_large_image_overlay.png')




