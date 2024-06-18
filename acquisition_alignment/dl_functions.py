import os
import pyrosm
import geopandas as gpd

# Function for Downloading
def download_osm(city_name, pyrosm_path):
    print(f"Downloading OSM data for {city_name}...")

    # Download the data into specified directory
    fp = pyrosm.get_data(f"{city_name}", directory=pyrosm_path)
    print(f"{city_name} data was downloaded to:", fp)

def get_building_bbox(pyrosm_path, city_name):
    if city_name == "Berlin":
        bbox = [13.294333, 52.454927, 13.500205, 52.574409]
        # Initialize the OSM object
    else:
        bbox = None
    
    fp = pyrosm.get_data(f"{city_name}", directory=pyrosm_path)
    print(fp)

    # Initialize the OSM object 
    osm = pyrosm.OSM(fp, bounding_box=bbox)
    
    # Retrieve building data
    buildings = osm.get_buildings()
    print("OSM data get_buildings done.")

    # Convert to GeoDataFrame
    buildings_gdf = gpd.GeoDataFrame(buildings, geometry='geometry', crs="EPSG:4326")
    buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    print(f"OSM for {city_name} converted into GeoDataFrame, with crs=EPSG:4326 done.")

    # Retrieve boundaries
    boundaries = osm.get_boundaries()
    minx, miny, maxx, maxy = boundaries.total_bounds      

    north, south, west, east = maxy, miny, minx, maxx
    print("OSM data get_boundaries done.")
    
    return [buildings_gdf, [north, south, west, east]]



#### Function for Downloading Sentinel-2 L2A Data
def download_sentinel2_images_openeo(connection, bbox, dates_interval, cloud_cover_percentage, output_path):
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
        bands = ["B04", "B03", "B02", "B08"], # Red, Green, Blue, Infrared (nir) bands
        max_cloud_cover = cloud_cover_percentage
    )

    result = datacube.save_result("GTiff")
    
    # Creating a new job at the back-end by sending the datacube information.
    job = result.create_job()
    
    # Starts the job and waits until it finished to download the result.
    job.start_and_wait()
    job.get_results().download_files(output_path)
    
    print("Sentinel-2 images downloaded successfully.")