from rasterio.features import geometry_mask
import numpy as np

def cloud_classifier(patch, threshold=0.4, cloud_fraction_limit=0.05):
    # Example: Use the mean of the blue band to classify clouds
    blue_band = patch[:, :, 2]
    cloud_fraction = np.mean(blue_band > threshold)
    
    return cloud_fraction < cloud_fraction_limit

# Create a binary mask of the buildings
def create_building_mask(buildings_gdf, transform, width, height):
    buildings_mask = geometry_mask(
        [geom for geom in buildings_gdf.geometry],
        transform=transform,
        invert=True,
        out_shape=(height, width)
    )
    return buildings_mask.astype(np.uint8)

def extract_patches(image, mask, patch_size):
    patches_image = []
    patches_mask = []
    # start: 0 -> patch_size, black margin on borders?....
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch_img = image[i:i + patch_size, j:j + patch_size, :]
            patch_msk = mask[i:i + patch_size, j:j + patch_size]

            if cloud_classifier(patch_img):
                patches_image.append(patch_img)
                patches_mask.append(patch_msk)
        
    patches_image = np.stack(patches_image)
    patches_mask = np.stack(patches_mask)
    
    return patches_image, patches_mask