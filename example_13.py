from os.path import dirname, abspath, exists, join
from typing import Sequence
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
FONT_COLOR: tuple = (25, 25, 25)
FONT_SCALE: float = 0.75
FONT_THICKNESS: int = 2


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
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def calculate_distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param point1: First point (x, y).
    :type point1: tuple[float, float]
    :param point2: Second point (x, y).
    :type point2: tuple[float, float]

    :return: Distance between the two points.
    :rtype: float
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def draw_line_with_distance(frame: np.array, point1: Sequence, point2: Sequence) -> None:
    """
    Draws a line between two points on an image frame, calculates the Euclidean distance
    between them, and displays the distance in centimeters near the midpoint of the line.
    This function modifies the input frame by overlaying the line and the distance text.

    :param frame: The image where the line and distance will be drawn and displayed.
    :type frame: np.array
    :param point1: The coordinates of the starting point of the line.
    :type point1: Sequence
    :param point2: The coordinates of the ending point of the line.
    :type point2: Sequence

    :return: This function does not return any value but modifies the frame directly.
    :rtype: None
    """
    cv2.line(frame, point1, point2, (0, 0, 0), 2)

    distance = np.linalg.norm(np.array(point1) - np.array(point2))
    distance_cm = (distance / MARKER_SIZE) * 100

    mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    cv2.putText(img=frame,
                text=f"{distance_cm:.2f} cm",
                org=mid_point,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=FONT_SCALE,
                color=FONT_COLOR,
                thickness=FONT_THICKNESS)


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))
    matrix, coefficients = camera_calibration(current_path=current_file_path)
    detector = aruco_detector()
    marker_colors = {}
    saved_positions = {}
    gray_template = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[INFO] Place ArUco markers in front of the camera.")
    print("[INFO] Press 'q' to quit, 'm' to save marker position, 'd' to clear positions.")

    while True:
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        if gray_template is None:
            gray_template = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_template)
        corners, ids, _ = detector.detectMarkers(gray_template)

        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                corner = corners[i][0].astype(int)

                if marker_id not in marker_colors:
                    marker_colors[marker_id] = tuple(np.random.randint(0, 255, 3).tolist())

                color = marker_colors[marker_id]
                cv2.polylines(frame, [corner], True, color, 2)

                current_pos = tuple(np.mean(corner, axis=0).astype(int))

                if marker_id in saved_positions:
                    saved_pos = saved_positions[marker_id]
                    cv2.circle(frame, saved_pos, 5, color, -1)
                    cv2.circle(frame, current_pos, 5, color, -1)

                    draw_line_with_distance(frame, saved_pos, current_pos)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m') and ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                corner = corners[i][0].astype(int)
                saved_positions[marker_id] = tuple(np.mean(corner, axis=0).astype(int))
                print(f"[INFO] Save marker {marker_id} position.")

        elif key == ord('d'):
            saved_positions.clear()
            print("[INFO] All saved positions cleared.")

        cv2.imshow("AR Marker Detection: pose estimation and marker tracking", frame)

    cap.release()
    cv2.destroyAllWindows()
