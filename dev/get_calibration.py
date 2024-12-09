from os.path import dirname, abspath, join
import cv2
import numpy as np


PATTERN: tuple = (9, 6)
SQUARE_SIZE: float = 0.024
FILE_PATH: str = "../src/camera_params.npz"


if __name__ == "__main__":
    current_file_path = dirname(abspath(__file__))

    obj_points = []
    img_points = []

    objp = np.zeros((np.prod(PATTERN), 3), dtype=np.float32)
    objp[:, :2] = np.indices(PATTERN).T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    cap = cv2.VideoCapture(0)
    print("[INFO] Press 's' to save the camera calibration parameters.")
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, PATTERN, None)

        if ret:
            img_points.append(corners)
            obj_points.append(objp)
            cv2.drawChessboardCorners(frame, PATTERN, corners, ret)

        cv2.imshow("Camera calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if len(obj_points) > 0 and len(img_points) > 0:
                _, matrix, coefficient, _, _ = cv2.calibrateCamera(obj_points,
                                                                   img_points,
                                                                   gray.shape[::-1],
                                                                   None,
                                                                   None)
                np.savez(join(current_file_path, FILE_PATH),
                         camera_matrix=matrix.astype(int),
                         dist_coefficients=np.round(coefficient, 2))

                print("[INFO] Matrix:\n", matrix.astype(int))
                print("[INFO] Coefficient:\n", np.round(coefficient, 2))
            else:
                print("[ERROR] Not enough data to calibrate the camera!")

    cap.release()
    cv2.destroyAllWindows()
