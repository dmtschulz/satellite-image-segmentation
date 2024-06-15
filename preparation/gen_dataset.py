import os
import glob
import numpy as np
from PIL import Image
from acquisition_alignment.plt_functions import stretch_contrast
from sklearn.model_selection import train_test_split
import shutil

def save_patches_and_masks(tensor_image, tensor_mask, output_dir, city, prefix='patch'):
    # Create directories if they don't exist
    image_dir = os.path.join(output_dir, 'all_images')
    mask_dir = os.path.join(output_dir, 'all_masks')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    num_patches = tensor_image.shape[0]
    
    for i in range(num_patches):
        # Convert the tensor image and mask to numpy arrays
        img = tensor_image[i].numpy()
        mask = tensor_mask[i].numpy()
        
        # Save the image patch
        img_stretched = np.zeros_like(img)
        for c in range(img.shape[2]):
            img_stretched[:, :, c] = stretch_contrast(img[:, :, c])
        
        # Normalize the stretched image for saving
        img_stretched = (img_stretched - np.min(img_stretched)) / (np.max(img_stretched) - np.min(img_stretched)) * 255
        img_stretched = img_stretched.astype(np.uint8)
        
        # Convert to PIL image and save
        img_pil = Image.fromarray(img_stretched[:, :, :3])  # Save only RGB channels
        img_pil.save(os.path.join(image_dir, f'{prefix}_{i+1}_{city}.png'))
        
        # Save the mask
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert mask to binary image
        mask_pil.save(os.path.join(mask_dir, f'{prefix}_mask_{i+1}_{city}.png'))
        
    print(f'Saved {num_patches} image patches and masks to {output_dir}')

# Function to pair image and mask files based on file naming convention
def pair_files(image_dir, mask_dir):
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    
    paired_files = []
    for img_file in image_files:
        base_name = os.path.basename(img_file).replace("patch_", "").replace(".png", "")
        corresponding_mask = os.path.join(mask_dir, f"patch_mask_{base_name}.png")
        if corresponding_mask in mask_files:
            paired_files.append((img_file, corresponding_mask))
    
    return paired_files

def split_and_save(paired_files, output_dir, test_split=0.2, val_split=0.2, random_state=42):
    # Create output directories
    train_img_dir = os.path.join(output_dir, 'train/images')
    train_mask_dir = os.path.join(output_dir, 'train/masks')
    val_img_dir = os.path.join(output_dir, 'val/images')
    val_mask_dir = os.path.join(output_dir, 'val/masks')
    test_img_dir = os.path.join(output_dir, 'test/images')
    test_mask_dir = os.path.join(output_dir, 'test/masks')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)
    
    # Split the data
    train_val_files, test_files = train_test_split(paired_files, test_size=test_split, random_state=random_state)
    train_files, val_files = train_test_split(train_val_files, test_size=val_split, random_state=random_state)
    
    # Function to copy files to respective directories
    def copy_files(file_pairs, img_dir, mask_dir):
        for img_file, mask_file in file_pairs:
            shutil.copy(img_file, img_dir)
            shutil.copy(mask_file, mask_dir)
    
    # Copy the files to respective directories
    copy_files(train_files, train_img_dir, train_mask_dir)
    copy_files(val_files, val_img_dir, val_mask_dir)
    copy_files(test_files, test_img_dir, test_mask_dir)
    
    print(f"Saved {len(train_files)} train, {len(val_files)} validation, and {len(test_files)} test file pairs.")
    