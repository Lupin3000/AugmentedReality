import cv2


ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50


if __name__ == "__main__":
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(frame_gray)

        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("AR Marker ID Detection: show id", frame)

    cap.release()
    cv2.destroyAllWindows()
