from datapull import main as datapull_main
from waterrefine import refined_water_data
import cv2

def main():
    # Adjust the values here to the lake you want to analyze
    # The default values are for the summer of 2022 between june and august with Google data

#lake Hopatcong with no contamination
    #latitude = 40.950065  
    #longitude = -74.634119
    #zoom = 2000

#round valley with contamination
    latitude = 40.615888
    longitude = -74.825174
    zoom = 2000

    datapull_main(latitude, longitude, zoom)
    
    refined_overlay, refined_algae_area, refined_lake_area = refined_water_data()

    if refined_lake_area > 0:
        contamination_percentage = (refined_algae_area / refined_lake_area) * 100
        print(f"Contamination Level: {contamination_percentage:.2f}%")
    else:
        print("Lake data error")

    if refined_overlay is not None:
        cv2.imshow("Algae Bloom", refined_overlay)
        cv2.imwrite("algae_bloom_overlay.jpg", refined_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No overlay to display.")

if __name__ == "__main__":
    main()