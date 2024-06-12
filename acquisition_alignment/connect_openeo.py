import openeo

#### Function for Connecting to OpenEO Copernicus
def connect_to_openeo():
    print("Connecting to OpenEO...")
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()
    print("Connected to OpenEO.\n")
    return connection