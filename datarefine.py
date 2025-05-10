import geemap
import os
from PIL import Image
import ee
import math

def calculate_tiles_needed(zoom):
    base_zoom = 2000  # base stuff for 9 images
    base_tiles = 9  # base stuff that spilts to 9 tiles
    tiles_needed = (zoom / base_zoom) ** 2 * base_tiles
    return math.ceil(tiles_needed) # always round up

def export_full_image(image, region):
    tif_dir = os.path.join('tif_water_data')
    jpg_dir = os.path.join('water_data')
    
    os.makedirs(tif_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    full_tif_path = os.path.join(tif_dir, 'water_full.tif')
    full_jpg_path = os.path.join(jpg_dir, 'water_full.jpg')

    visualization = {
        'bands': ['B4', 'B3', 'B2'],  # RGB bands
        'min': 0.0,
        'max': 0.3,
    }

    geemap.ee_export_image(
        image.visualize(**visualization),
        filename=full_tif_path,
        scale=10,  # regular scale for Sentinel-2 (the satellite's native resolution)
        region=region,
        file_per_band=False
    )

    if os.path.exists(full_tif_path):
        with Image.open(full_tif_path) as img:
            rgb_img = img.convert('RGB')
            rgb_img.save(full_jpg_path, 'JPEG')
    else:
        print(f"Error: Full zoomed out tif file not found at {full_tif_path}")

    return full_tif_path, full_jpg_path

def export_tiles(image, region, zoom, num_tiles):

    bounds = region.bounds().getInfo()
    coordinates = bounds['coordinates'][0] 
    x_min, y_min = coordinates[0]  # Bottom left 
    x_max, y_max = coordinates[2]  # Top right 

    x_split = (x_max - x_min) / math.sqrt(num_tiles)
    y_split = (y_max - y_min) / math.sqrt(num_tiles)

    def create_tile(x_start, y_start, x_end, y_end):
        return ee.Geometry.Rectangle([x_start, y_start, x_end, y_end])

    tif_dir = os.path.join('tif_water_data')
    jpg_dir = os.path.join('water_data')
    os.makedirs(tif_dir, exist_ok=True)
    os.makedirs(jpg_dir, exist_ok=True)

    for i in range(int(math.sqrt(num_tiles))):
        for j in range(int(math.sqrt(num_tiles))):
            x_start = x_min + i * x_split
            y_start = y_min + j * y_split
            x_end = x_min + (i + 1) * x_split
            y_end = y_min + (j + 1) * y_split

            tile_region = create_tile(x_start, y_start, x_end, y_end)
            tif_path = os.path.join(tif_dir, f'water_tile_{i}_{j}.tif')
            jpg_path = os.path.join(jpg_dir, f'water_tile_{i}_{j}.jpg')

            geemap.ee_export_image(
                image.visualize(bands=['B4', 'B3', 'B2'], min=0.0, max=0.3),
                filename=tif_path,
                scale=10,
                region=tile_region,
                file_per_band=False
            )

            #google outputs it as a tif file so i convert it to jpg
            if os.path.exists(tif_path):
                with Image.open(tif_path) as img:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(jpg_path, 'JPEG')
            else:
                print(f"Error: TIF file not found at {tif_path}")
