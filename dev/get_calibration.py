from os.path import dirname, abspath, join
from sys import exit
from threading import Thread
import numpy as np
import cv2


WINDOW_WIDTH: int = 1152
WINDOW_HEIGHT: int = 720
FPS: int = 30

PATTERN: tuple = (9, 6)
SQUARE_SIZE: float = 0.024
FILE_PATH: str = "../src/camera_params.npz"


def calibrate_camera(object_points: np.array, image_points: list, gray_shape: tuple, path: str) -> None:
    """
    Calibrates a camera using object points and image points, calculates the camera matrix and
    distortion coefficients, and saves them to a specified file path.

    :param object_points: A NumPy array of 3D points in the real world.
    :type object_points: np.array
    :param image_points: A list of 2D points in the image plane corresponding to the object points.
    :type image_points: list
    :param gray_shape: A tuple representing the shape of the grayscale image.
    :type gray_shape: tuple
    :param path: A string representing the file path template to save the calibration results.
    :type path: str

    :return: None
    """
    try:
        print("[INFO] Starting calibration...")
        c_ret, matrix, coefficients, _, _ = cv2.calibrateCamera(objectPoints=object_points,
                                                                imagePoints=image_points,
                                                                imageSize=gray_shape[::-1],
                                                                cameraMatrix=None,
                                                                distCoeffs=None)
        if c_ret:
            np.savez(file=path,
                     camera_matrix=matrix.astype(np.float32),
                     dist_coefficients=coefficients.astype(np.float32))

            print("[INFO] Calibration successful!")
            print(f"[INFO] Saved calibration parameters to: {path}")
            print(f"[INFO] Camera matrix:\n{matrix}")
            print(f"[INFO] Distortion coefficients:\n{coefficients}")
        else:
            print("[ERROR] Calibration failed.")
    except Exception as e:
        print(f"[ERROR] Calibration error: {e}")


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))
    gray_template = None

    obj_points = []
    img_points = []

    objp = np.zeros((np.prod(PATTERN), 3), dtype=np.float32)
    objp[:, :2] = np.indices(PATTERN).T.reshape(-1, 2) * SQUARE_SIZE

    calibration_thread = None
    collected_frames = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Error opening video stream.")
        exit(1)
    else:
        print("[INFO] Place chessboard pattern in front of the camera.")
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
        found, corners = cv2.findChessboardCorners(gray_template, PATTERN, None)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not (calibration_thread and calibration_thread.is_alive()):
            if collected_frames >= 10:
                output_path = join(current_file_path, FILE_PATH)
                calibration_thread = Thread(target=calibrate_camera,
                                            args=(obj_points, img_points, gray_template.shape, output_path))
                calibration_thread.start()
            else:
                print("[ERROR] Not enough valid data to calibrate the camera! Try to capture more frames.")

        if found:
            cv2.drawChessboardCorners(frame, PATTERN, corners, found)

            if collected_frames < 20:
                img_points.append(corners)
                obj_points.append(objp)
                collected_frames += 1

                print(f"[INFO] Collected frame {collected_frames}/20.")

        cv2.imshow("Camera calibration via chessboard pattern", frame)

    cap.release()
    cv2.destroyAllWindows()

    if calibration_thread and calibration_thread.is_alive():
        calibration_thread.join()
