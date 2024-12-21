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
ERASE_SIZE: int = 20


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
    detector = aruco_detector()

    canvas = None
    prev_pos = None
    mode = None

    cap = cv2.VideoCapture(0)
    print("[INFO] Place ArUco markers in front of the camera.")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        if canvas is None:
            canvas = np.zeros_like(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for i, corner_group in enumerate(corners):
                marker_id = int(ids[i][0])
                center = tuple(np.mean(corner_group[0], axis=0).astype(int))

                if marker_id == 0:
                    mode = 'erase'
                    cv2.circle(frame, center, ERASE_SIZE, PEN_COLOR[0], 2)

                    if prev_pos is not None:
                        cv2.line(canvas, prev_pos, center, ERASE_COLOR, ERASE_SIZE)

                if 1 <= marker_id < len(PEN_COLOR):
                    mode = 'draw'
                    cv2.circle(frame, center, PEN_SIZE, PEN_COLOR[marker_id], -1)

                    if prev_pos is not None:
                        cv2.line(canvas, prev_pos, center, PEN_COLOR[marker_id], PEN_SIZE)

                prev_pos = center

        frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow("AR Marker ID Detection: draw on screen", frame)

    cap.release()
    cv2.destroyAllWindows()
