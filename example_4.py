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
INFO_COLOR_A: tuple = (150, 150, 150)
INFO_COLOR_B: tuple = (150, 200, 200)


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
        params = np.load(param_file)
        return params["camera_matrix"].astype(np.float32), params["dist_coefficients"].astype(np.float32)
    else:
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

        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if len(ids) > 1:
                centers = []
                for i in range(len(corners)):
                    c = corners[i][0]
                    center = c.mean(axis=0)
                    centers.append(center)

                cv2.line(frame, tuple(map(int, centers[0])), tuple(map(int, centers[1])), INFO_COLOR_A, 2)

                pt1 = tuple(map(int, centers[0]))
                pt2 = tuple(map(int, centers[1]))
                midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

                distance_pixels = np.linalg.norm(np.array(pt1) - np.array(pt2))
                message = f"Distance: {distance_pixels:.2f} px"

                cv2.putText(frame, message, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_COLOR_A, 2)

                ret_1, _, vec_1 = cv2.solvePnP(OBJ_POINTS, corners[0], matrix, coefficients)
                ret_2, _, vec_2 = cv2.solvePnP(OBJ_POINTS, corners[1], matrix, coefficients)

                if ret_1 and ret_2:
                    midpoint_below = (midpoint[0], midpoint[1] + 20)

                    distance_meters = np.linalg.norm(vec_1 - vec_2)
                    distance_cm = distance_meters * 100

                    message = f"Distance: {distance_cm:.2f} cm"
                    cv2.putText(frame, message, midpoint_below, cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_COLOR_B, 2)


        cv2.imshow("AR Marker ID Detection: Draw line", frame)

    cap.release()
    cv2.destroyAllWindows()
