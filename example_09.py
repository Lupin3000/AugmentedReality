from os.path import dirname, abspath, exists, join
from sys import exit
import cv2
import numpy as np


WINDOW_WIDTH: int = 1152
WINDOW_HEIGHT: int = 720
FPS: int = 30

ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50

EXAMPLE_PATH: str = "src/photos/"


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
    example_path = join(current_file_path, EXAMPLE_PATH)
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

        if ids is not None and len(ids) > 1:
            marker_id_sum = int(ids[0][0] + ids[1][0])
            img_path = join(example_path, f"treasure_{marker_id_sum}.jpg")

            if not exists(img_path):
                print(f"[ERROR] Image not found: {img_path}")
                continue

            if marker_id_sum not in image_cache:
                print(f"[INFO] Loading image: {img_path}")
                image_cache[marker_id_sum] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            image_capture = image_cache[marker_id_sum]

            corners_2 = corners[0][0]
            corners_1 = corners[1][0]

            frame = draw_image_between_markers(frame, corners_1, corners_2, image_capture)

        cv2.imshow("AR Marker Detection: show image on two markers", frame)

    cap.release()
    cv2.destroyAllWindows()
