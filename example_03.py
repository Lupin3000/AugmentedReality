import cv2
import numpy as np


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

    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


if __name__ == "__main__":
    detector = aruco_detector()

    cap = cv2.VideoCapture(0)
    print("[INFO] Place ArUco markers in front of the camera.")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

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
