import cv2
import numpy as np


def main():
    video_path = 'Lane Detection Test Video 01.mp4'
    cam = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cam.read()
        old_shape = frame.shape

        ratio = old_shape[0] / old_shape[1]

        # Exercise 2
        width = 420
        frame = cv2.resize(frame, (int(width), int(width * ratio)))

        # Exercise 3
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height = frame.shape[0]
        width = frame.shape[1]

        # Exercise 4
        upper_left = (int(width * 0.33), int(height * 0.75))
        upper_right = (int(width * 0.6), int(height * 0.75))
        lower_left = (int(0), int(height - 1))
        lower_right = (int(width - 1), int(height - 1))

        points_trapezoid = np.array([upper_left, upper_right, lower_right, lower_left], dtype='int32')

        frame_trapezoid = np.zeros((height, width), dtype='uint8')
        cv2.fillConvexPoly(frame_trapezoid, points=points_trapezoid, color=1)
        frame_trapezoid = frame_trapezoid * frame
        # Exercise 5
        screen_points = np.array([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)], dtype='float32')
        magical_matrix = cv2.getPerspectiveTransform(np.float32(points_trapezoid), screen_points)

        stretched_frame_trapezoid = cv2.warpPerspective(frame_trapezoid, magical_matrix, (width, height))

        # Exercise 6
        frame = cv2.blur(stretched_frame_trapezoid, ksize=(3, 3))

        # Exercise 7
        sobel_vertical = np.float32([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        sobel_horizontal = np.transpose(sobel_vertical)

        frame_f = np.float32(frame)

        frame_1 = cv2.filter2D(frame_f, -1, sobel_vertical)
        frame_2 = cv2.filter2D(frame_f, -1, sobel_horizontal)

        # frame_int = cv2.convertScaleAbs(frame_2)
        combined = np.sqrt(frame_1 * frame_1 + frame_2 * frame_2)

        frame = cv2.convertScaleAbs(combined)

        # Exercise 8
        threshold = int(150)

        frame = np.array(frame > threshold, dtype='uint8')
        frame = frame * 255

        # Exercise 9
        copy_frame = frame.copy()
        nr = int(width * 0.2)
        copy_frame[0:width, 0:nr] = 0
        copy_frame[0:width, (width - nr): width] = 0

        left_xs = []
        left_ys = []
        right_xs = []
        right_ys = []

        half = int(width / 2)
        first_half = copy_frame[0:width, 0:half]
        second_half = copy_frame[0:width, half:width]

        left_points = np.argwhere(first_half > 1)
        right_points = np.argwhere(second_half > 1)

        for i in range(left_points.shape[0]):
            for j in range(left_points.shape[1]):
                left_xs.append(j)
                left_ys.append(i)

        for i in range(right_points.shape[0]):
            for j in range(right_points.shape[1]):
                right_xs.append(j)
                right_ys.append(i)

        frame = copy_frame


        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()