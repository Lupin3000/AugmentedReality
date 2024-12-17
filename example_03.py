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
FONT_COLOR: tuple = (50, 50, 50)
FONT_SCALE: float = 1.0
FONT_THICKNESS: int = 2
FONT_FACE: int = cv2.FONT_HERSHEY_SIMPLEX
ARROW_COLOR: tuple = (10, 255, 10)
ARROW_THICKNESS: int = 3


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
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 1:
            marker_centers = []

            for corner_group in corners:
                top_left = corner_group[0][0]
                bottom_right = corner_group[0][2]
                center_x = int((top_left[0] + bottom_right[0]) / 2)
                center_y = int((top_left[1] + bottom_right[1]) / 2)
                marker_centers.append((center_x, center_y))

            for i, (center_x, center_y) in enumerate(marker_centers):
                for j, (other_x, other_y) in enumerate(marker_centers):
                    if i != j:
                        cv2.arrowedLine(img=frame,
                                        pt1=(center_x, center_y),
                                        pt2=(other_x, other_y),
                                        color=ARROW_COLOR,
                                        thickness=ARROW_THICKNESS,
                                        line_type=cv2.LINE_AA)

                        distance = int(np.sqrt((other_x - center_x) ** 2 + (other_y - center_y) ** 2))
                        mid_x = (center_x + other_x) // 2
                        mid_y = (center_y + other_y) // 2

                        cv2.putText(img=frame,
                                    text=f"{distance} px",
                                    org=(mid_x, mid_y),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=FONT_SCALE,
                                    color=FONT_COLOR,
                                    thickness=FONT_THICKNESS,
                                    lineType=cv2.LINE_AA)

        cv2.imshow("AR Marker ID Detection: show arrows and distance", frame)

    cap.release()
    cv2.destroyAllWindows()
