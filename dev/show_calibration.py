from os.path import dirname, abspath, join, exists
import numpy as np


FILE_PATH: str = "../src/camera_params.npz"


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))

    if exists(join(current_file_path, FILE_PATH)):
        params = np.load(join(current_file_path, FILE_PATH))
        matrix = params["camera_matrix"]
        coefficients = params["dist_coefficients"]

        print("[INFO] Matrix:\n", matrix)
        print("[INFO] Coefficient:\n", coefficients)
    else:
        print("[ERROR] Camera calibration parameters not found!")