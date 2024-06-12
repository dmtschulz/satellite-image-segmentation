import os
import pyrosm
import geopandas as gpd


pyrosm_path = "./acquisition_alignment/pyrosm_cities"
# Ensure the directory exists
os.makedirs(pyrosm_path, exist_ok=True)

#### Function for Downloading and Processing OSM Data
def download_and_process_osm_data(city_name):
    print(f"Downloading and processing OSM data for {city_name}...")

    # Download the data into specified directory
    fp = pyrosm.get_data(f"{city_name}", directory=pyrosm_path)
    print(f"{city_name} data was downloaded to:", fp)

    # Initialize the OSM object 
    osm = pyrosm.OSM(fp)

    # Retrieve building data
    buildings = osm.get_buildings()
    print("OSM data get_buildings done.")

    # Retrieve boundaries
    boundaries = osm.get_boundaries()
    minx, miny, maxx, maxy = boundaries.total_bounds
    north, south, west, east = maxy, miny, minx, maxx
    print("OSM data get_boundaries done.")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(buildings, geometry='geometry', crs="EPSG:4326")
    print(f"OSM for {city_name} converted into GeoDataFrame, with crs=EPSG:4326 done.")

    return gdf, [north, south, west, east]



#### Function for Downloading Sentinel-2 L2A Data
def download_sentinel2_images_openeo(connection, bbox, dates_interval, cloud_cover_percentage, city):
    print("Downloading Sentinel-2 L2a images from OpenEO...")

    # Define the area of interest
    spatial_extent = {
        "north": bbox[0],
        "south": bbox[1],
        "west": bbox[2],
        "east": bbox[3]
    }
    
    # Define the process graph
    datacube = connection.load_collection(
        collection_id = "SENTINEL2_L2A",
        spatial_extent = spatial_extent,
        temporal_extent = dates_interval,
        bands = ["B04", "B03", "B02", "B08"] # Red, Green, Blue, Infrared (nir) bands
        # max_cloud_cover = cloud_cover_percentage
    )

    result = datacube.save_result("GTiff")
    
    # Creating a new job at the back-end by sending the datacube information.
    job = result.create_job()
    
    # Starts the job and waits until it finished to download the result.
    job.start_and_wait()
    job.get_results().download_files(f"openeo_cities/{city}/")
    
    print("Sentinel-2 images downloaded successfully.")