import cv2
import numpy as np


ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50
SMALL_RECT_COLOR: tuple = (80, 25, 200)
BIG_RECT_COLOR: tuple = (200, 25, 80)
MERGE_DISTANCE: int =250


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


def calculate_distance(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param point1: First point (x, y).
    :type point1: tuple[float, float]
    :param point2: Second point (x, y).
    :type point2: tuple[float, float]

    :return: Distance between the two points.
    :rtype: float
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


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

        if ids is not None:
            merged_markers = []
            distances = {}
            centers = []

            for marker_corners in corners:
                top_left = tuple(marker_corners[0][0].astype(int))
                bottom_right = tuple(marker_corners[0][2].astype(int))
                center_x = (top_left[0] + bottom_right[0]) // 2
                center_y = (top_left[1] + bottom_right[1]) // 2
                centers.append((center_x, center_y))

            for i, center1 in enumerate(centers):
                for j, center2 in enumerate(centers):
                    if i < j:
                        distances[(i, j)] = calculate_distance(center1, center2)

            merged_indices = set()
            for (i, j), dist in distances.items():
                if dist < MERGE_DISTANCE:
                    merged_indices.add(i)
                    merged_indices.add(j)

            if merged_indices:
                min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
                for idx in merged_indices:
                    marker_corners = corners[idx]
                    top_left = tuple(marker_corners[0][0].astype(int))
                    top_right = tuple(marker_corners[0][1].astype(int))
                    bottom_right = tuple(marker_corners[0][2].astype(int))
                    bottom_left = tuple(marker_corners[0][3].astype(int))

                    min_x = min(min_x, top_left[0], bottom_left[0])
                    max_x = max(max_x, top_right[0], bottom_right[0])
                    min_y = min(min_y, top_left[1], top_right[1])
                    max_y = max(max_y, bottom_left[1], bottom_right[1])

                cv2.rectangle(img=frame,
                              pt1=(min_x, min_y),
                              pt2=(max_x, max_y),
                              color=BIG_RECT_COLOR,
                              thickness=-1)

            for i, marker_corners in enumerate(corners):
                if i not in merged_indices:
                    top_left = tuple(marker_corners[0][0].astype(int))
                    top_right = tuple(marker_corners[0][1].astype(int))
                    bottom_right = tuple(marker_corners[0][2].astype(int))
                    bottom_left = tuple(marker_corners[0][3].astype(int))

                    min_x = min(top_left[0], bottom_left[0])
                    max_x = max(top_right[0], bottom_right[0])
                    min_y = min(top_left[1], top_right[1])
                    max_y = max(bottom_left[1], bottom_right[1])

                    cv2.rectangle(img=frame,
                                  pt1=(min_x, min_y),
                                  pt2=(max_x, max_y),
                                  color=SMALL_RECT_COLOR,
                                  thickness=-1)

        cv2.imshow("AR Marker ID Detection: show rectangles and merge them below specific distance", frame)

    cap.release()
    cv2.destroyAllWindows()
