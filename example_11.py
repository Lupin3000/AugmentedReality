from sys import exit
from typing import Sequence
import cv2
import numpy as np


ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50
RECT_COLOR: tuple = (200, 25, 25)
MERGE_DISTANCE: int = 250


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


def calculate_center(corners: np.ndarray) -> tuple[int, int]:
    """
    Calculate the center point of an ArUco marker.

    :param corners: The corners of the marker.
    :type corners: np.ndarray

    :return: The center coordinates (x, y).
    :rtype: tuple[int, int]
    """
    top_left = tuple(corners[0][0].astype(int))
    bottom_right = tuple(corners[0][2].astype(int))
    center_x = (top_left[0] + bottom_right[0]) // 2
    center_y = (top_left[1] + bottom_right[1]) // 2
    return int(center_x), int(center_y)


def draw_rectangle(frame: np.ndarray, corners: np.ndarray) -> None:
    """
    Draw a rectangle on the frame using the corners of the ArUco marker.

    :param frame: The image on which to draw the rectangle.
    :type frame: np.ndarray
    :param corners: The corners of the marker to draw the rectangle around.
    :type corners: np.ndarray

    :return: None
    """
    top_left = tuple(corners[0][0].astype(int))
    top_right = tuple(corners[0][1].astype(int))
    bottom_right = tuple(corners[0][2].astype(int))
    bottom_left = tuple(corners[0][3].astype(int))

    min_x = min(top_left[0], bottom_left[0])
    max_x = max(top_right[0], bottom_right[0])
    min_y = min(top_left[1], top_right[1])
    max_y = max(bottom_left[1], bottom_right[1])

    cv2.rectangle(img=frame,
                  pt1=(min_x, min_y),
                  pt2=(max_x, max_y),
                  color=RECT_COLOR,
                  thickness=-1)


def merge_markers(corners: Sequence[np.ndarray], merge_distance: int) -> list[list[int]]:
    """
    Merge markers if their centers are within a specified distance.

    :param corners: List of marker corners.
    :type corners: Sequence[np.ndarray]
    :param merge_distance: The distance threshold for merging markers.
    :type merge_distance: int

    :return: A list of lists, each containing indices of markers that belong to the same merged group.
    :rtype: list[list[int]]
    """
    centers = [calculate_center(marker_corners) for marker_corners in corners]
    n = len(centers)
    clusters = []

    assigned = [False] * n

    for i in range(n):
        if not assigned[i]:
            cluster = [i]
            assigned[i] = True

            for j in range(i + 1, n):
                if not assigned[j] and calculate_distance(centers[i], centers[j]) < merge_distance:
                    cluster.append(j)
                    assigned[j] = True

            clusters.append(cluster)

    return clusters


if __name__ == "__main__":
    detector = aruco_detector()
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

        if gray_template is None:
            gray_template = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_template)
        corners, ids, _ = detector.detectMarkers(gray_template)

        if ids is not None:
            marker_clusters = merge_markers(corners, MERGE_DISTANCE)

            for cluster in marker_clusters:
                min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
                for idx in cluster:
                    marker_corners = corners[idx]
                    mc_top_left = tuple(marker_corners[0][0].astype(int))
                    mc_top_right = tuple(marker_corners[0][1].astype(int))
                    mc_bottom_right = tuple(marker_corners[0][2].astype(int))
                    mc_bottom_left = tuple(marker_corners[0][3].astype(int))

                    min_x = min(min_x, mc_top_left[0], mc_bottom_left[0])
                    max_x = max(max_x, mc_top_right[0], mc_bottom_right[0])
                    min_y = min(min_y, mc_top_left[1], mc_top_right[1])
                    max_y = max(max_y, mc_bottom_left[1], mc_bottom_right[1])

                cv2.rectangle(img=frame,
                              pt1=(min_x, min_y),
                              pt2=(max_x, max_y),
                              color=RECT_COLOR,
                              thickness=-1)

            for i, marker_corners in enumerate(corners):
                if not any(i in cluster for cluster in marker_clusters):
                    draw_rectangle(frame, marker_corners)

        cv2.imshow("AR Marker ID Detection: show rectangles and merge them below specific distance", frame)

    cap.release()
    cv2.destroyAllWindows()
