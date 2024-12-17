from os.path import dirname, abspath, exists, join
from itertools import combinations
import cv2
import numpy as np


MARKER_SIZE: float = 0.035
ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50
OBJ_POINTS: np.ndarray = np.array([
        [0, 0, 0],
        [MARKER_SIZE, 0, 0],
        [MARKER_SIZE, MARKER_SIZE, 0],
        [0, MARKER_SIZE, 0]
    ], dtype=np.float32)
FILE_PARAMS_PATH: str = "src/camera_params.npz"
FONT_COLOR: tuple = (50, 50, 50)
FONT_SCALE: float = 1.0
FONT_THICKNESS: int = 2
FONT_FACE: int = cv2.FONT_HERSHEY_SIMPLEX
LINE_COLOR: tuple = (25, 255, 25)
LINE_THICKNESS: int = 2


def camera_calibration(current_path: str) -> tuple:
    """
    Performs camera calibration by loading camera matrix and distortion
    coefficients from a specified file path. If the file does not exist,
    it returns default intrinsic parameters and zero distortion coefficients.

    :param current_path: File path where camera parameters file is located.
    :type current_path: str

    :return: A tuple containing the camera matrix and distortion coefficients.
    :rtype: tuple
    """
    param_file = join(current_path, FILE_PARAMS_PATH)

    if exists(param_file):
        print(f"[INFO] Loading camera parameters from: {param_file}")
        params = np.load(param_file)
        return params["camera_matrix"].astype(np.float32), params["dist_coefficients"].astype(np.float32)
    else:
        print("[INFO] Camera parameters file not found. Using default values.")
        return np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32), np.zeros(5)


def aruco_detector() -> cv2.aruco.ArucoDetector:
    """
    Initializes and returns an ArUco detector configured with a predefined
    dictionary and default detection parameters.

    :return: A configured ArUcoDetector instance ready to detect markers.
    :rtype: cv2.aruco.ArucoDetector
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()

    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def calculate_distance(tvec_1: np.ndarray, tvec_2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two translation vectors.
    The return is float value in centimeters.

    :param tvec_1: Translation vector of marker 1.
    :type tvec_1: np.ndarray
    :param tvec_2: Translation vector of marker 2.
    :type tvec_2: np.ndarray

    :return: Distance in centimeters.
    :rtype: float
    """
    tvec_1 = np.array(tvec_1).flatten()
    tvec_2 = np.array(tvec_2).flatten()
    distance_meters = np.linalg.norm(tvec_1 - tvec_2)

    return distance_meters * 100


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))

    matrix, coefficients = camera_calibration(current_path=current_file_path)
    detector = aruco_detector()

    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(frame_gray)

        tvecs = []
        centers = []

        if ids is not None:
            for i in range(len(ids)):
                retval, rvec, tvec = cv2.solvePnP(OBJ_POINTS, corners[i][0], matrix, coefficients)
                tvecs.append(tvec)

                center_x = int(np.mean(corners[i][0][:, 0]))
                center_y = int(np.mean(corners[i][0][:, 1]))
                centers.append((center_x, center_y))

            if len(tvecs) > 1:
                for (idx1, idx2) in combinations(range(len(tvecs)), 2):
                    distance = calculate_distance(tvecs[idx1], tvecs[idx2])
                    center_1 = centers[idx1]
                    center_2 = centers[idx2]

                    cv2.line(frame, center_1, center_2, LINE_COLOR, LINE_THICKNESS)

                    mid_point = (int((center_1[0] + center_2[0]) / 2), int((center_1[1] + center_2[1]) / 2))

                    cv2.putText(img=frame,
                                text=f"{distance:.2f} cm",
                                org=mid_point,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=FONT_SCALE,
                                color=FONT_COLOR,
                                thickness=FONT_THICKNESS,
                                lineType=cv2.LINE_AA)

        cv2.imshow("AR Marker ID Detection: pose estimation and distance", frame)

    cap.release()
    cv2.destroyAllWindows()
