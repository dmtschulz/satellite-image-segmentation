from rasterio.features import geometry_mask
import numpy as np
import torch

def pixel_filter(mask_patch, pixel_fraction_limit=0.0002):
    pixel_fraction = np.mean(mask_patch == 1)
    
    return pixel_fraction < pixel_fraction_limit

def cloud_classifier(patch, threshold=0.8, cloud_fraction_limit=0.05):
    # Access the blue band (index 2)
    blue_band = patch[:, :, 2]
    cloud_fraction = np.mean(blue_band > threshold)
    
    return cloud_fraction < cloud_fraction_limit

def extract_patches(image, mask, patch_size):
    patches_image = []
    patches_mask = []
    
    height, width, _ = image.shape
    
    for i in range(patch_size, height - patch_size + 1, patch_size):
        for j in range(patch_size, width - patch_size + 1, patch_size):
            patch_msk = mask[i:i + patch_size, j:j + patch_size]

            if pixel_filter(patch_msk): # skip too black masks
                continue

            patch_img = image[i:i + patch_size, j:j + patch_size, :]

            if cloud_classifier(patch_img):
                patches_image.append(patch_img)
                patches_mask.append(patch_msk)
        
    patches_image = np.stack(patches_image)
    patches_mask = np.stack(patches_mask)
    
    # Convert to torch tensors
    tensor_image = torch.tensor(patches_image, dtype=torch.float32)
    tensor_mask = torch.tensor(patches_mask, dtype=torch.long)
    
    return tensor_image, tensor_mask