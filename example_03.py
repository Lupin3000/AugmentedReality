from sys import exit
import cv2
import numpy as np


WINDOW_WIDTH: int = 1152
WINDOW_HEIGHT: int = 720
FPS: int = 30

ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50

FONT_COLOR: tuple = (50, 50, 50)
FONT_SCALE: float = 1.0
FONT_THICKNESS: int = 2
FONT_FACE: int = cv2.FONT_HERSHEY_SIMPLEX
ARROW_COLOR: tuple = (10, 255, 10)
ARROW_THICKNESS: int = 3


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

        cv2.imshow("AR Marker ID Detection: show arrows and distance between markers", frame)

    cap.release()
    cv2.destroyAllWindows()
