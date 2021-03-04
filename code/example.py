"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture("1.mp4")

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    h_ratio=ratio = gaze.horizontal_ratio()
    v_ratio=ratio = gaze.vertical_ratio()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "h ratio: " + str(h_ratio), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "v ratio:  " + str(v_ratio), (90, 235), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    print(str(h_ratio),type(h_ratio))

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
