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


def draw_image_between_markers(img: np.ndarray,
                               corners_marker_0: np.ndarray,
                               corners_marker_1: np.ndarray,
                               overlay_image: np.ndarray) -> np.ndarray:
    """
    Draws an overlay image between two markers on a given image using a homography
    transformation.

    :param img: The target image on which overlay image will be drawn
    :type img: np.ndarray
    :param corners_marker_0: Corner points of the first marker (marker 0)
    :type corners_marker_0: np.ndarray
    :param corners_marker_1: Corner points of the second marker (marker 1)
    :type corners_marker_1: np.ndarray
    :param overlay_image: The overlay image to be drawn between the markers
    :type overlay_image: np.ndarray

    :return: The image with the overlay drawn between the markers
    :rtype: np.ndarray
    """
    top_left_corner = corners_marker_0[np.argmin(corners_marker_0.sum(axis=1))]
    bottom_right_corner = corners_marker_1[np.argmax(corners_marker_1.sum(axis=1))]

    overlay_width = int(np.linalg.norm(top_left_corner[0] - bottom_right_corner[0]))
    # overlay_height = int(overlay_width * (overlay_image.shape[0] / overlay_image.shape[1]))

    dest_points = np.array([
        top_left_corner,
        [top_left_corner[0] + overlay_width, top_left_corner[1]],
        bottom_right_corner,
        [bottom_right_corner[0] - overlay_width, bottom_right_corner[1]]
    ], dtype=np.float32)

    src_points = np.array([
        [0, 0],
        [overlay_image.shape[1], 0],
        [overlay_image.shape[1], overlay_image.shape[0]],
        [0, overlay_image.shape[0]]
    ], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(src_points, dest_points)
    if homography_matrix is None:
        print("[WARNING] Homography matrix is None. Returning original image.")
        return img

    warped_overlay = cv2.warpPerspective(overlay_image, homography_matrix, (img.shape[1], img.shape[0]))
    if warped_overlay.shape[2] == 4:
        alpha_channel = warped_overlay[:, :, 3] / 255.0
        rgb_overlay = warped_overlay[:, :, :3]

        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - alpha_channel) + rgb_overlay[:, :, c] * alpha_channel
    else:
        mask = (warped_overlay > 0).any(axis=2)
        img[mask] = warped_overlay[mask]

    return img


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))

    matrix, coefficients = camera_calibration(current_path=current_file_path)
    detector = aruco_detector()

    cap = cv2.VideoCapture(0)
    image_cache = {}
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) >= 2:
            marker_indices = {mid[0]: idx for idx, mid in enumerate(ids)}

            if 0 in marker_indices and 1 in marker_indices:
                idx1 = marker_indices[0]
                idx2 = marker_indices[1]

                marker_id = int(idx1 + idx2)
                img_path = join(current_file_path, f"src/photos/treasure_{marker_id}.jpg")

                if not exists(img_path):
                    print(f"[ERROR] Image not found: {img_path}")
                    continue

                if marker_id not in image_cache:
                    print(f"[INFO] Loading image: {img_path}")
                    image_cache[marker_id] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                image_capture = image_cache[marker_id]

                corners1 = corners[idx1][0]
                corners2 = corners[idx2][0]

                frame = draw_image_between_markers(frame, corners1, corners2, image_capture)

        cv2.imshow("AR Multi-Marker Detection: with image", frame)

    cap.release()
    cv2.destroyAllWindows()
