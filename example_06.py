from os.path import dirname, abspath, exists, join
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


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))

    matrix, coefficients = camera_calibration(current_path=current_file_path)
    detector = aruco_detector()

    cap = cv2.VideoCapture(0)
    print("[INFO] Place ArUco markers in front of the camera.")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, corner_group in enumerate(corners):
                m_ret, rvec, tvec = cv2.solvePnP(OBJ_POINTS, corner_group, matrix, coefficients)

                if m_ret:
                    frame = cv2.drawFrameAxes(frame, matrix, coefficients, rvec, tvec, 0.05)

        cv2.imshow("AR Marker ID Detection: pose estimation and show X, Y and Z coordination axes", frame)

    cap.release()
    cv2.destroyAllWindows()
