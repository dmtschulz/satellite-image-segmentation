import os
import pyrosm
import geopandas as gpd

#### Function for Downloading and Processing OSM Data
def download_and_process_osm_data(city_name, pyrosm_path):
    print(f"Downloading and processing OSM data for {city_name}...")

    # Download the data into specified directory
    fp = pyrosm.get_data(f"{city_name}", directory=pyrosm_path)
    print(f"{city_name} data was downloaded to:", fp)

    # Initialize the OSM object 
    osm = pyrosm.OSM(fp)

    # Retrieve building data
    buildings = osm.get_buildings()
    print("OSM data get_buildings done.")

    # Convert to GeoDataFrame
    buildings_gdf = gpd.GeoDataFrame(buildings, geometry='geometry', crs="EPSG:4326")
    # buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    print(f"OSM for {city_name} converted into GeoDataFrame, with crs=EPSG:4326 done.")

    # Retrieve boundaries
    boundaries = osm.get_boundaries()
    minx, miny, maxx, maxy = boundaries.total_bounds
    
    # Calculate the center of the bounding box
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    # Determine the size of the square bounding box
    width = maxx - minx
    height = maxy - miny
    max_dimension = max(width, height)

    # Define the new bounds for the square bounding box
    minx_square = center_x - max_dimension / 2
    maxx_square = center_x + max_dimension / 2
    miny_square = center_y - max_dimension / 2
    maxy_square = center_y + max_dimension / 2        

    north, south, west, east = maxy_square, miny_square, minx_square, maxx_square
    print("OSM data get_boundaries done.")

    #buildings_gdf.to_file(pyrosm_path+f'buildings_gdf_{city_name}.shp')
    #print("Buildings GeoDataFrame saved to a .shp file.")

    #with open(pyrosm_path+f"boundaries_{city_name}", "w") as file:
    #    file.write([north, south, west, east])
    #print("Boundaries saved to a shp file.")
    
    return buildings_gdf, [north, south, west, east]



#### Function for Downloading Sentinel-2 L2A Data
def download_sentinel2_images_openeo(connection, bbox, dates_interval, cloud_cover_percentage, city, output_path):
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
    job.get_results().download_files(output_path+city)
    
    print("Sentinel-2 images downloaded successfully.")