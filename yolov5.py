#!/usr/bin/env python3.6
# encoding: utf-8
import cv2 as cv
from yolov5_trt import YoLov5TRT
file_yaml = 'digital_number.yaml'

if __name__ == "__main__":
    capture = cv.VideoCapture(0)
    capture.set(6, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(file_yaml)
    while capture.isOpened():
        ret, frame = capture.read()
        if cv.waitKey(1) & 0xFF == ord('q'): break
        frame, use_time = yolov5_wrapper.infer(frame)
        fps = 1.0 / use_time
        text = "FPS : " + str(int(fps))
        cv.putText(frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        cv.imshow('frame', frame)
    capture.release()
    cv.destroyAllWindows()
    # destroy the instance
    yolov5_wrapper.destroy()
