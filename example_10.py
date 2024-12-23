from os.path import dirname, abspath, exists, join
from sys import exit
import cv2
import numpy as np


ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50
OBJECT_COLOR: tuple = (255, 255, 255)
OBJECT_THICKNESS: int = -1
EXAMPLE_PATH: str = "src/videos/"


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


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))
    example_path = join(current_file_path, EXAMPLE_PATH)
    video_path = join(example_path, "demo.mp4")

    if not exists(video_path):
        print(f"[INFO] Video file {video_path} not found.")
        exit()
    else:
        print(f"[INFO] Using video file: {video_path}")

    video_cap = cv2.VideoCapture(video_path)
    detector = aruco_detector()
    gray_template = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[INFO] Place ArUco markers in front of the camera.")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        if gray_template is None:
            gray_template = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_template)
        corners, ids, _ = detector.detectMarkers(gray_template)

        video_ret, video_frame = video_cap.read()
        if not video_ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_ret, video_frame = video_cap.read()

        if ids is not None:
            all_points = np.concatenate(corners, axis=0).reshape(-1, 2)

            center, radius = cv2.minEnclosingCircle(all_points)
            center = tuple(map(int, center))
            radius = int(radius)

            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.circle(img=mask, center=center, radius=radius, color=OBJECT_COLOR, thickness=OBJECT_THICKNESS)

            video_frame = cv2.resize(video_frame, (radius * 2, radius * 2))
            video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)

            v_h, v_w, _ = video_frame.shape
            x_start, y_start = center[0] - v_w // 2, center[1] - v_h // 2
            x_end, y_end = x_start + v_w, y_start + v_h

            if x_start < 0 or y_start < 0 or x_end > frame.shape[1] or y_end > frame.shape[0]:
                continue

            roi = frame[y_start:y_end, x_start:x_end]
            masked_video = cv2.bitwise_and(video_frame, video_frame, mask=mask[y_start:y_end, x_start:x_end, 0])
            masked_frame = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask[y_start:y_end, x_start:x_end, 0]))
            frame[y_start:y_end, x_start:x_end] = cv2.add(masked_video, masked_frame)

        cv2.imshow("AR Marker ID Detection: all markers create a video file mask", frame)

    cap.release()
    cv2.destroyAllWindows()
