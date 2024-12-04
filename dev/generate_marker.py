from os import makedirs
from os.path import dirname, abspath, join, isdir
import cv2


ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_ID: int = 3
ARUCO_MARKER_SIZE: int = 100


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))
    target_directory = join(current_file_path, "img/markers")
    target_file_path = join(target_directory, f"marker_{ARUCO_MARKER_ID}.jpg")

    if not isdir(target_directory):
        print(f"[INFO] Create directory: {target_directory}")
        makedirs(target_directory)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict=ARUCO_DICT_ID)
    marker = cv2.aruco.generateImageMarker(dictionary=aruco_dict, id=ARUCO_MARKER_ID, sidePixels=ARUCO_MARKER_SIZE)

    print(f"[INFO] Marker is saved: {target_file_path}")
    cv2.imwrite(filename=target_file_path, img=marker)
