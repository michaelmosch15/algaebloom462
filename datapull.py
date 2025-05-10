import os
import ee
from dotenv import load_dotenv
from datarefine import calculate_tiles_needed, export_full_image, export_tiles

load_dotenv()

def main(lat, lon, zoom):
    project_id = os.getenv('PROJECT_ID')
    if not project_id:
        raise ValueError("Problem with the env file")

    ee.Initialize(project=project_id)

    water_region = ee.Geometry.Point(lon, lat)
    region = water_region.buffer(zoom).bounds()

    dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate('2022-06-01', '2022-08-31') \
        .filterBounds(water_region) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  
    
    def add_s2clouds(img):
        clouds = ee.Image(img.get('cloud_mask')).select('probability')
        return img.addBands(clouds.rename('cloud_prob'))

    s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
        .filterBounds(water_region) \
        .filterDate('2022-06-01', '2022-08-31')

    dataset = ee.ImageCollection(ee.Join.saveFirst('cloud_mask').apply(
        primary=dataset,
        secondary=s2_clouds,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    )).map(add_s2clouds)

    def mask_s2_clouds(img):
        cloud_mask = img.select('cloud_prob').lt(40)  
        return img.updateMask(cloud_mask).clip(region)

    dataset = dataset.map(mask_s2_clouds)

    def apply_scale_factors(image):
        optical_bands = image.select(['B2', 'B3', 'B4']).multiply(0.0001)
        return image.addBands(optical_bands, overwrite=True)

    dataset = dataset.map(apply_scale_factors)

    image = ee.Image(dataset.first()).clip(region)
    
    #tile stuff
    num_tiles = calculate_tiles_needed(zoom)  
    print(f"Number of tiles for zoom level {zoom}: {num_tiles}")

    full_tif_path, full_jpg_path = export_full_image(image, region)

    export_tiles(image, region, zoom, num_tiles)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("error with internal arg")
