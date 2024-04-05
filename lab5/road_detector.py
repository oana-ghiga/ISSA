import cv2
import numpy as np

class RoadDetector:
    def __init__(self, video_path):
        self.cam = cv2.VideoCapture(video_path)
        self.pts = None

    def convert_to_gray(self, frame):
        """
                Converts a given frame to grayscale.

                Parameters:
                - frame (numpy.ndarray): The input frame (RGB or BGR format).

                Returns:
                - gray_frame (numpy.ndarray): The frame converted to grayscale.
                """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def create_mask(self, frame):
        """
               Creates a mask for the road region in the frame.

               Parameters:
               - frame (numpy.ndarray): The input frame (grayscale).

               Returns:
               - mask (numpy.ndarray): The binary mask indicating the road region.
               """
        mask = np.zeros_like(frame)
        height, width = frame.shape
        # Define points for perspective transformation
        pt1 = (int(width * 0.57), int(height * 0.77))  # top right
        pt2 = (int(width * 0.45), int(height * 0.77))  # top left
        pt3 = (0, height)  # bottom left
        pt4 = (width, height)  # bottom right
        self.pts = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
        cv2.fillConvexPoly(mask, self.pts, 1)
        return mask

    def apply_mask(self, frame, mask):
        """
                Applies a binary mask to the frame.

                Parameters:
                - frame (numpy.ndarray): The input frame (grayscale).
                - mask (numpy.ndarray): The binary mask.

                Returns:
                - masked_frame (numpy.ndarray): The frame with the mask applied.
                """
        return cv2.multiply(frame, mask, dtype=cv2.CV_8U)

    def display_small(self, frame, window_name, size):
        """
               Displays a given frame in a window of specified size.

               Parameters:
               - frame (numpy.ndarray): The input frame.
               - window_name (str): The name of the window.
               - size (tuple): The size of the displayed frame (width, height).
               """
        frame = cv2.resize(frame, size)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cam.release()
            cv2.destroyAllWindows()

    def get_perspective_transform(self, frame):
        """
               Calculates the perspective transform matrix for a frame.

               Parameters:
               - frame (numpy.ndarray): The input frame.

               Returns:
               - M (numpy.ndarray): The perspective transform matrix.
               """
        src_points = np.float32(self.pts)
        height, width = frame.shape
        dst_points = np.float32([(width, 0), (0, 0), (0, height), (width, height)])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return M

    def apply_perspective_transform(self, frame, M):
        """
               Applies a perspective transform to the frame.

               Parameters:
               - frame (numpy.ndarray): The input frame.
               - M (numpy.ndarray): The perspective transform matrix.

               Returns:
               - warped (numpy.ndarray): The frame with the perspective transform applied.
               """
        height, width = frame.shape
        warped = cv2.warpPerspective(frame, M, (width, height))
        return warped

    def apply_blur(self, frame, ksize):
        """
               Applies Gaussian blur to a frame. Gaussian blur is used to reduce noise in the image, it works by averaging the pixel values in the neighborhood of each pixel.

               Parameters:
               - frame (numpy.ndarray): The input frame.
               - ksize (tuple): Kernel size for Gaussian blur (width, height).

               Returns:
               - blurred_frame (numpy.ndarray): The blurred frame.
               """
        return cv2.blur(frame, ksize)

    def apply_sobel(self, frame, ksize):
        """
                Applies Sobel edge detection to a frame. Sobel edge detection is a technique used to detect edges in an image by calculating the gradient of the image intensity.

                Parameters:
                - frame (numpy.ndarray): The input frame.
                - ksize (int): Kernel size for Sobel operator.

                Returns:
                - sobel_frame (numpy.ndarray): The frame with Sobel edges.
                """
        frame = np.float32(frame)
        sobel_v = np.float32(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]])
        sobel_h = np.transpose(sobel_v)
        grad_v = cv2.filter2D(frame, -1, sobel_v)
        grad_h = cv2.filter2D(frame, -1, sobel_h)
        grad = np.sqrt(grad_v ** 2 + grad_h ** 2)
        grad = cv2.convertScaleAbs(grad)
        return grad

    def binarize_frame(self, frame, threshold):
        """
               Binarizes a frame based on a threshold.

               Parameters:
               - frame (numpy.ndarray): The input frame.
               - threshold (int): Threshold value for binarization.

               Returns:
               - binary_frame (numpy.ndarray): The binarized frame.
               """
        _, binary_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        return binary_frame

    def get_lane_markings(self, frame, threshold):
        """
               Extracts lane markings from a binary frame.

               Parameters:
               - frame (numpy.ndarray): The input binary frame.
               - threshold (int): Threshold value for lane detection.

               Returns:
               - left_x (numpy.ndarray): X-coordinates of left lane markings.
               - left_y (numpy.ndarray): Y-coordinates of left lane markings.
               - right_x (numpy.ndarray): X-coordinates of right lane markings.
               - right_y (numpy.ndarray): Y-coordinates of right lane markings.
               """
        height, width = frame.shape
        frame[:, :int(width * 0.03)] = 0
        frame[:, int(width * 0.97):] = 0
        frame[int(height * 0.97):, :] = 0
        _, binary_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
        left_half = binary_frame[:, :width // 2]
        right_half = binary_frame[:, width // 2:]
        left_coordinates = np.argwhere(left_half == 255)
        right_coordinates = np.argwhere(right_half == 255)
        left_y, left_x = left_coordinates[:, 0], left_coordinates[:, 1]
        right_y, right_x = right_coordinates[:, 0], right_coordinates[:, 1] + width // 2
        return left_x, left_y, right_x, right_y

    def get_lane_lines(self, left_x, left_y, right_x, right_y):
        """
                Fits lane lines to extracted lane markings.

                Parameters:
                - left_x (numpy.ndarray): X-coordinates of left lane markings.
                - left_y (numpy.ndarray): Y-coordinates of left lane markings.
                - right_x (numpy.ndarray): X-coordinates of right lane markings.
                - right_y (numpy.ndarray): Y-coordinates of right lane markings.

                Returns:
                - left_line (numpy.ndarray): Coefficients of the left lane line.
                - right_line (numpy.ndarray): Coefficients of the right lane line.
                """
        left_line = np.polynomial.polynomial.polyfit(left_y, left_x, deg=1)
        right_line = np.polynomial.polynomial.polyfit(right_y, right_x, deg=1)
        return left_line, right_line

    def draw_birds_eye_view_lane_lines(self, frame, left_line, right_line):
        """
                Draws lane lines on a top-down view of the road.

                Parameters:
                - frame (numpy.ndarray): The top-down view frame.
                - left_line (numpy.ndarray): Coefficients of the left lane line.
                - right_line (numpy.ndarray): Coefficients of the right lane line.

                Returns:
                - frame (numpy.ndarray): The frame with drawn lane lines.
                """
        height, _ = frame.shape
        y = np.array([0, height])
        left_x = left_line[1] * y + left_line[0]
        right_x = right_line[1] * y + right_line[0]
        left_line_pts = np.float32([[left_x[0], y[0]], [left_x[1], y[1]]])
        right_line_pts = np.float32([[right_x[0], y[0]], [right_x[1], y[1]]])
        cv2.line(frame, (int(left_line_pts[0][0]), int(left_line_pts[0][1])),
                 (int(left_line_pts[1][0]), int(left_line_pts[1][1])), (200, 0, 0), 5)
        cv2.line(frame, (int(right_line_pts[0][0]), int(right_line_pts[0][1])),
                 (int(right_line_pts[1][0]), int(right_line_pts[1][1])), (100, 0, 0), 5)
        return frame

    def draw_lane_lines(self, frame, left_line, right_line):
        """
                Draws lane lines on the original frame.

                Parameters:
                - frame (numpy.ndarray): The original frame.
                - left_line (numpy.ndarray): Coefficients of the left lane line.
                - right_line (numpy.ndarray): Coefficients of the right lane line.

                Returns:
                - frame (numpy.ndarray): The frame with drawn lane lines.
                """
        height, _, _ = frame.shape
        y = np.array([0, height])
        left_x = left_line[1] * y + left_line[0]
        right_x = right_line[1] * y + right_line[0]
        left_line_pts = np.float32([[left_x[0], y[0]], [left_x[1], y[1]]])
        right_line_pts = np.float32([[right_x[0], y[0]], [right_x[1], y[1]]])
        M_inv = cv2.getPerspectiveTransform(np.float32([(frame.shape[1], 0), (0, 0), (0, frame.shape[0]), (frame.shape[1], frame.shape[0])]), np.float32(self.pts))
        left_line_pts = cv2.perspectiveTransform(np.array([left_line_pts]), M_inv)[0]
        right_line_pts = cv2.perspectiveTransform(np.array([right_line_pts]), M_inv)[0]
        cv2.line(frame, (int(left_line_pts[0][0]), int(left_line_pts[0][1])),
                 (int(left_line_pts[1][0]), int(left_line_pts[1][1])), (50, 50, 250), 5)
        cv2.line(frame, (int(right_line_pts[0][0]), int(right_line_pts[0][1])),
                 (int(right_line_pts[1][0]), int(right_line_pts[1][1])), (50, 250, 50), 5)
        return frame

    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                print("End of video")
                break
            gray = self.convert_to_gray(frame)
            self.display_small(gray, "1.gray", (320, 240))
            mask = self.create_mask(gray)
            self.display_small(mask, "2.mask", (320, 240))
            road = self.apply_mask(gray, mask)
            self.display_small(road, "3.road", (320, 240))
            M = self.get_perspective_transform(road)
            top_down_view = self.apply_perspective_transform(road, M)
            self.display_small(top_down_view, "4.top_down_view", (320, 240))
            blurred_view = self.apply_blur(top_down_view, (3, 3))
            self.display_small(blurred_view, "5.blurred_view", (320, 240))
            edge_view = self.apply_sobel(blurred_view, 3)
            self.display_small(edge_view, "6.edge_view", (320, 240))
            binary_view = self.binarize_frame(edge_view, 80)
            self.display_small(binary_view, "7.binary_view", (320, 240))
            left_x, left_y, right_x, right_y = self.get_lane_markings(binary_view, 200)
            if left_x.size > 0 and right_x.size > 0:
                left_line, right_line = self.get_lane_lines(left_x, left_y, right_x, right_y)
                frame_with_birds_eye_view_lanes = self.draw_birds_eye_view_lane_lines(top_down_view, left_line, right_line)
                self.display_small(frame_with_birds_eye_view_lanes, "8.frame_with_birds_eye_view_lanes", (320, 240))
                frame_with_lanes = self.draw_lane_lines(frame, left_line, right_line)
                cv2.imshow('Lane Lines', frame_with_lanes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.cam.release()
                    cv2.destroyAllWindows()

road_detector = RoadDetector('Lane Detection Test Video 01.mp4')
road_detector.run()
