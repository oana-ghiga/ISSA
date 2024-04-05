import cv2
import object_socket
import numpy as np

class RoadDetector:
    def __init__(self, ip='127.0.0.1', port=5000):
        self.receiver = object_socket.ObjectReceiverSocket(ip, port)
        self.pts = None

    def convert_to_gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def create_mask(self, frame):
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
        return cv2.multiply(frame, mask, dtype=cv2.CV_8U)

    def display_small(self, frame, window_name, size):
        frame = cv2.resize(frame, size)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def get_perspective_transform(self, frame):
        src_points = np.float32(self.pts)
        height, width = frame.shape
        dst_points = np.float32([(width, 0), (0, 0), (0, height), (width, height)])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return M

    def apply_perspective_transform(self, frame, M):
        height, width = frame.shape
        warped = cv2.warpPerspective(frame, M, (width, height))
        return warped

    def apply_blur(self, frame, ksize):
        return cv2.blur(frame, ksize)

    def apply_sobel(self, frame, ksize):
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
        _, binary_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        return binary_frame

    def get_lane_markings(self, frame, threshold):
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
        left_line = np.polynomial.polynomial.polyfit(left_y, left_x, deg=1)
        right_line = np.polynomial.polynomial.polyfit(right_y, right_x, deg=1)
        return left_line, right_line

    def draw_birds_eye_view_lane_lines(self, frame, left_line, right_line):
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

    def process_frame(self, frame):
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
                cv2.destroyAllWindows()

    def run(self):
        while True:
            frame = self.receiver.recv_object()
            if frame is None:
                break

            print("Received a frame from producer")
            self.process_frame(frame)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = RoadDetector()
    detector.run()
