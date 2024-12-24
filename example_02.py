from sys import exit
import cv2
import numpy as np


WINDOW_WIDTH: int = 1152
WINDOW_HEIGHT: int = 720
FPS: int = 30

ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50

FONT_COLOR: tuple = (100, 200, 200)
FONT_SCALE: float = 5.0
FONT_THICKNESS: int = 5
FONT_FACE: int = cv2.FONT_HERSHEY_SIMPLEX


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


def id_to_letter(m_id: int) -> str:
    """
    Converts a numerical marker ID to a corresponding letter (A-Z).
    If the ID exceeds the alphabet range, it wraps around using modulo.

    :param m_id: The numerical marker ID.
    :type m_id: int

    :return: Corresponding letter as a string.
    :rtype: str
    """
    alphabet_size = 26
    return chr((int(m_id) % alphabet_size) + ord('A'))


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

        if ids is not None:
            for i, corner_group in enumerate(corners):
                marker_id = int(ids[i][0])
                letter = id_to_letter(marker_id)

                top_left = corner_group[0][0]
                bottom_right = corner_group[0][2]
                center_x = int((top_left[0] + bottom_right[0]) / 2)
                center_y = int((top_left[1] + bottom_right[1]) / 2)

                cv2.putText(img=frame,
                            text=letter,
                            org=(center_x, center_y),
                            fontFace=FONT_FACE,
                            fontScale=FONT_SCALE,
                            color=FONT_COLOR,
                            thickness=FONT_THICKNESS,
                            lineType=cv2.LINE_AA)

        cv2.imshow("AR Marker ID Detection: show fonts on each marker", frame)

    cap.release()
    cv2.destroyAllWindows()
