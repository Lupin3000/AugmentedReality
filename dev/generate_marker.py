from os import makedirs
from os.path import dirname, abspath, join, isdir
from argparse import ArgumentParser
import cv2


ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_ID: int = 0
ARUCO_MARKER_SIZE: int = 100


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))
    target_directory = join(current_file_path, "img/markers")

    if not isdir(target_directory):
        print(f"[INFO] Create directory: {target_directory}")
        makedirs(target_directory)

    parser = ArgumentParser(description="Generate ArUco marker JPG")
    parser.add_argument("--id", type=int, default=ARUCO_MARKER_ID, help="ArUco marker id")
    parser.add_argument("--size", type=int, default=ARUCO_MARKER_SIZE, help="ArUco marker size in pixels")
    args = parser.parse_args()

    target_file_path = join(target_directory, f"marker_{args.id}.jpg")

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict=ARUCO_DICT_ID)
    marker = cv2.aruco.generateImageMarker(dictionary=aruco_dict, id=args.id, sidePixels=args.size)

    print(f"[INFO] Marker ID {args.id} with size {args.size} is saved: {target_file_path}")
    cv2.imwrite(filename=target_file_path, img=marker)
