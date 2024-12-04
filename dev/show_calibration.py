from os.path import dirname, abspath, join
import numpy as np


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))

    params = np.load(join(current_file_path, "../src/camera_params.npz"))
    matrix = params["camera_matrix"]
    coefficients = params["dist_coefficients"]

    print("[INFO] Matrix:\n", matrix)
    print("[INFO] Coefficient:\n", coefficients)