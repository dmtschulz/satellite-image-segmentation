import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np

#### Adjust brightness for pictures
def stretch_contrast(band, low_percent=2, high_percent=98):
    """
    Stretch the contrast of a given band using percentile clipping.
    """
    in_min = np.percentile(band, low_percent)
    in_max = np.percentile(band, high_percent)
    out_min, out_max = 0.0, 1.0
    
    band_stretched = (band - in_min) / (in_max - in_min)
    band_stretched[band_stretched < out_min] = out_min
    band_stretched[band_stretched > out_max] = out_max
    return band_stretched

#### Function for plotting Buildings from OpenStreetMaps
def plot_city_buildings(city_name, buildings):
    
    # Plot the building footprints
    fig, ax = plt.subplots(figsize=(10, 10))
    buildings.plot(ax=ax, color='blue', edgecolor='none', alpha=0.7)

    # Set title and labels
    plt.title(f"Buildings in {city_name}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Ensure the directory exists
    image_dir = f"images/{city_name}/"
    os.makedirs(image_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(image_dir+"OSM_buildings.png", bbox_inches='tight', pad_inches=0)
    
    plt.show()

### Save RGB Image
def rgb_image(sentinel2_path, output_image_path):
    # Load the Sentinel-2 image
    with rasterio.open(sentinel2_path) as src:
        # Read the Red, Green, and Blue bands by their indices (B04, B03, B02)
        red = src.read(1)  # B04
        green = src.read(2)  # B03
        blue = src.read(3)  # B02

        # Stretch the contrast of each band
        red_stretched = stretch_contrast(red)
        green_stretched = stretch_contrast(green)
        blue_stretched = stretch_contrast(blue)

        # Stack the bands into an RGB image
        rgb = np.dstack((red_stretched, green_stretched, blue_stretched))

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb) # Render the image array
    plt.axis('off')  # Turn off axis
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.show()  # Show the image
    plt.close()

### Save Single Band Image
def single_band_image(sentinel2_path, output_image_path, band_index):
    # Load the Sentinel-2 image
    with rasterio.open(sentinel2_path) as src:
        # Read the specified band
        band = src.read(band_index)

        # Stretch the contrast of the band
        band_stretched = stretch_contrast(band)

    plt.figure(figsize=(10, 10))
    plt.imshow(band_stretched, cmap="gray") # Render the image array
    plt.axis('off')  # Turn off axis
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.show()  # Show the image
    plt.close()


# Define function to get CRS of the satellite image
def get_epsg_from_tif(tif_path):
    with rasterio.open(tif_path) as src:
        return src.crs.to_epsg()

# Define function to reproject GeoDataFrame
def reproject_gdf(gdf, target_crs):
    return gdf.to_crs(target_crs)

### Save Overlay Image
def overlay_image(buildings_gdf, sentinel2_path, output_image_path, band_index):
    # Get CRS of the satellite image
    satellite_crs = get_epsg_from_tif("openeo_cities/Tokyo/openEO_2021-06-11Z.tif")

    # Reproject buildings to match CRS of the satellite image
    gdf_projected = reproject_gdf(buildings_gdf, satellite_crs)

    # Load the satellite image
    with rasterio.open(sentinel2_path) as src:
        sentinel2_image = src.read(band_index, masked=True)  # Read the first band, considering any nodata value as masked
        # leave dark, dont strech...
        extent = rasterio.plot.plotting_extent(src)  # Get extent of the image

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the satellite image
    show(sentinel2_image, extent=extent, ax=ax, cmap='gray')

    # Plot the projected buildings
    gdf_projected.plot(ax=ax, color='blue', alpha=0.7)

    # Save
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)  # removes extra white space around figure

    # Display the figure
    plt.show()

### Save IRB Image
def irb_image(sentinel2_path, output_image_path):
    # Load the Sentinel-2 image
    with rasterio.open(sentinel2_path) as src:
        # Read the Red, Green, and Blue bands by their indices (B04, B03, B02)
        red = src.read(1)  # B04
        # green = src.read(2)  # B03
        blue = src.read(3)  # B02
        i = src.read(4)  # B08

        # Stretch the contrast of each band
        red_stretched = stretch_contrast(red)
        # green_stretched = stretch_contrast(green)
        blue_stretched = stretch_contrast(blue)
        i_stretched = stretch_contrast(i)

        # Stack the bands into an RGB image
        irb = np.dstack((i_stretched, red_stretched, blue_stretched))


    plt.figure(figsize=(10, 10))
    plt.imshow(irb) # Render the image array
    plt.axis('off')  # Turn off axis
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.show()  # Show the image
    plt.close()