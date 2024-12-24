from sys import exit
import cv2
import numpy as np


ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50
PEN_SIZE: int = 10
PEN_COLOR: dict = {
    3: (255, 0, 0),
    2: (0, 255, 0),
    1: (0, 0, 255),
    0: (255, 255, 255)
}
ERASE_COLOR: tuple = (0, 0, 0)
ERASE_SIZE: int = 25


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
    detector = aruco_detector()
    canvas = None
    prev_pos = None
    gray_template = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
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

        if canvas is None:
            canvas = np.zeros_like(frame, dtype=np.uint8)

        if gray_template is None:
            gray_template = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_template)
        corners, ids, _ = detector.detectMarkers(gray_template)

        if ids is not None and len(ids) > 0:
            for i, corner_group in enumerate(corners):
                marker_id = int(ids[i][0])
                center = tuple(np.mean(corner_group[0], axis=0).astype(int))

                if marker_id == 0:
                    cv2.circle(frame, center, ERASE_SIZE, PEN_COLOR[0], 2)

                    if prev_pos is not None:
                        cv2.line(canvas, prev_pos, center, ERASE_COLOR, ERASE_SIZE)

                if marker_id < len(PEN_COLOR):
                    cv2.circle(frame, center, PEN_SIZE, PEN_COLOR[marker_id], -1)

                    if prev_pos is not None:
                        cv2.line(canvas, prev_pos, center, PEN_COLOR[marker_id], PEN_SIZE)

                prev_pos = center
        else:
            prev_pos = None

        blended_frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0.0)
        cv2.imshow("AR Marker ID Detection: draw on screen", blended_frame)

    cap.release()
    cv2.destroyAllWindows()
