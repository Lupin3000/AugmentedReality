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
EXAMPLE_PATH: str = "src/videos/"


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


def draw_video_on_marker(img: np.ndarray,
                         rotation_vector: np.ndarray,
                         translation_vector: np.ndarray,
                         camera_matrix: np.ndarray,
                         dist_coefficients: np.ndarray,
                         video: cv2.VideoCapture) -> np.ndarray:
    """
    Draws a video frame onto a detected marker in the provided image.

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
    :param video: A cv2.VideoCapture object used to read frames from a video source.
    :type video: cv2.VideoCapture

    :return: An image with the video frame overlaid on the detected marker.
    :rtype: np.ndarray
    """
    v_ret, overlay_frame = video.read()
    if not v_ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        v_ret, overlay_frame = video.read()

    video_height, video_width = overlay_frame.shape[:2]
    video_aspect_ratio = video_width / video_height

    img_points, _ = cv2.projectPoints(OBJ_POINTS, rotation_vector, translation_vector, camera_matrix, dist_coefficients)
    img_points = np.int32(img_points).reshape(-1, 2)

    rect = cv2.boundingRect(img_points)
    x, y, marker_width, marker_height = rect
    new_width = int(marker_height * video_aspect_ratio)
    overlay_frame_resized = cv2.resize(overlay_frame, (new_width, marker_height))
    new_x = x + (marker_width - new_width) // 2
    new_x = max(0, new_x)
    overlay_frame_resized = overlay_frame_resized[:, :min(new_width, img.shape[1] - new_x)]

    for val in range(3):
        img[y:y + marker_height, new_x:new_x + overlay_frame_resized.shape[1], val] = overlay_frame_resized[:, :, val]

    return img

if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))
    example_path = join(current_file_path, EXAMPLE_PATH)

    matrix, coefficients = camera_calibration(current_path=current_file_path)
    detector = aruco_detector()

    video_cache = {}

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
            for i in range(len(ids)):
                marker_id = ids[i][0]
                video_path = join(example_path, f"video_{marker_id}.mp4")

                if not exists(video_path):
                    print(f"[ERROR] Video not found: {video_path}")
                    continue

                if marker_id not in video_cache:
                    print(f"[INFO] Loading video: {video_path}")
                    video_cache[marker_id] = cv2.VideoCapture(video_path)

                video_capture = video_cache[marker_id]
                raw_img_points = corners[i][0]

                m_ret, r_vec, t_vec = cv2.solvePnP(OBJ_POINTS, raw_img_points, matrix, coefficients)

                if m_ret:
                    frame = draw_video_on_marker(frame, r_vec, t_vec, matrix, coefficients, video_capture)

        cv2.imshow("AR Marker Detection: pose estimation and show video on each marker", frame)

    cap.release()
    for vc in video_cache.values():
        vc.release()
    cv2.destroyAllWindows()
