import rasterio
from rasterio.features import geometry_mask
import numpy as np


# Create a binary mask of the buildings
def create_building_mask(buildings_gdf, transform, width, height):
    buildings_mask = geometry_mask(
        [geom for geom in buildings_gdf.geometry],
        transform=transform,
        invert=True,
        out_shape=(height, width)
    )
    return buildings_mask.astype(np.uint8)

# Function to calculate building pixel ratio for a patch
def calculate_building_pixel_ratio(mask):
    total_pixels = mask.size
    building_pixels = np.sum(mask)
    return building_pixels / total_pixels


def cloud_classifier(patch, threshold=0.8, cloud_fraction_limit=0.05):
    """
    I used this function with normalized bands, thats why the threshold has this value.
    In the current version of the project, this function is not beeing used since
    I pass a cloud coverage parameter to the provider.
    """
    # Access the blue band (index 2)
    blue_band = patch[:, :, 2]
    cloud_fraction = np.mean(blue_band > threshold)
    
    return cloud_fraction < cloud_fraction_limit

# Function to create patches from an image and return as tensors
def create_patches(image_path, mask_path, patch_size, lo, hi):
    with rasterio.open(image_path) as src:
        img = src.read()
    
    with rasterio.open(mask_path) as src_mask:
        mask_img = src_mask.read(1)

    height, width = mask_img.shape

    img_patches = []
    mask_patches = []

    for i in range(patch_size, height, patch_size): # not from 0 since margin?
        for j in range(patch_size, width, patch_size): # not from 0 since gap
            img_patch = img[:, i:i+patch_size, j:j+patch_size]
            mask_patch = mask_img[i:i+patch_size, j:j+patch_size]

            # Check if the patch size is smaller than the defined patch size
            if img_patch.shape[1] < patch_size or img_patch.shape[2] < patch_size:
                continue
            # Do not make cloud classifier since images are already without clouds from provider.
            if lo <= calculate_building_pixel_ratio(mask_patch) <= hi:
                img_patches.append(img_patch)
                mask_patches.append(mask_patch)
            else:
                continue

    # Convert lists to numpy arrays
    img_patches = np.array(img_patches)
    mask_patches = np.array(mask_patches)

    return img_patches, mask_patches