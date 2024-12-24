from os.path import dirname, abspath, exists, join
from sys import exit
import cv2
import numpy as np


WINDOW_WIDTH: int = 1152
WINDOW_HEIGHT: int = 720
FPS: int = 30

MARKER_SIZE: float = 0.035
ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50
OBJ_POINTS: np.ndarray = np.array([
        [0, 0, 0],
        [MARKER_SIZE, 0, 0],
        [MARKER_SIZE, MARKER_SIZE, 0],
        [0, MARKER_SIZE, 0]
    ], dtype=np.float32)
FILE_PARAMS_PATH: str = "src/camera_params.npz"

EXAMPLE_PATH: str = "src/photos/"


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


def draw_image_on_marker(img: np.ndarray,
                         rotation_vector: np.ndarray,
                         translation_vector :np.ndarray,
                         camera_matrix: np.ndarray,
                         dist_coefficients: np.ndarray,
                         overlay_image: np.array) -> np.ndarray:
    """
    Draws a specified overlay image onto a detected marker within a given image.

    :param img: The input frame onto which the overlay will be drawn (BGR format).
    :type img: np.ndarray
    :param rotation_vector: The rotation vector that describes the orientation of the marker.
    :type rotation_vector: np.ndarray
    :param translation_vector: The translation vector that describes the position of the marker.
    :type translation_vector: np.ndarray
    :param camera_matrix: The intrinsic camera matrix for the camera.
    :type camera_matrix: np.ndarray
    :param dist_coefficients: The distortion coefficients of the camera.
    :type dist_coefficients: np.ndarray
    :param overlay_image: The image to overlay on the detected marker.
    :type overlay_image: np.ndarray

    :return: The modified image with the overlay image drawn on the detected marker.
    :rtype: np.ndarray
    """
    img_points, _ = cv2.projectPoints(OBJ_POINTS, rotation_vector, translation_vector, camera_matrix, dist_coefficients)
    img_points = np.int32(img_points).reshape(-1, 2)

    rect = cv2.boundingRect(img_points)
    x, y, w, h = rect
    overlay_image_resized = cv2.resize(overlay_image, (w, h))

    if overlay_image_resized.shape[2] == 4:
        overlay_image_resized_rgb = overlay_image_resized[:, :, :3]
        overlay_alpha = overlay_image_resized[:, :, 3:] / 255.0
        overlay_image_resized_rgb = (overlay_image_resized_rgb * overlay_alpha).astype(np.uint8)
    else:
        overlay_image_resized_rgb = overlay_image_resized

    for val in range(0, 3):
        img[y:y + h, x:x + w, val] = overlay_image_resized_rgb[:, :, val]

    return img


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))
    example_path = join(current_file_path, EXAMPLE_PATH)
    matrix, coefficients = camera_calibration(current_path=current_file_path)
    detector = aruco_detector()
    image_cache = {}
    gray_template = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Error opening video stream.")
        exit(1)
    else:
        print("[INFO] Place ArUco markers in front of the camera.")
        print("[INFO] Press 'q' or 'ESC' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        if frame is None or frame.size == 0:
            print("[WARNING] Empty frame. Skipping...")
            continue

        if gray_template is None:
            gray_template = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_template)
        corners, ids, _ = detector.detectMarkers(gray_template)

        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                img_path = join(example_path, f"monk_{marker_id}.jpg")

                if not exists(img_path):
                    print(f"[ERROR] Image not found: {img_path}")
                    continue

                if marker_id not in image_cache:
                    print(f"[INFO] Loading image: {img_path}")
                    image_cache[marker_id] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                image_capture = image_cache[marker_id]

                raw_img_points = corners[i][0]
                m_ret, r_vec, t_vec = cv2.solvePnP(OBJ_POINTS, raw_img_points, matrix, coefficients)

                if m_ret:
                    frame = draw_image_on_marker(frame, r_vec, t_vec, matrix, coefficients, image_capture)

        cv2.imshow("AR Marker Detection: pose estimation and show image on each marker", frame)

    cap.release()
    cv2.destroyAllWindows()
