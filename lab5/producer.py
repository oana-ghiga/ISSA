import cv2
import object_socket

class FrameProducer:
    def __init__(self, video_path, ip='127.0.0.1', port=5000):
        self.video_path = video_path
        self.cam = cv2.VideoCapture(video_path)
        self.sender = object_socket.ObjectSenderSocket(ip, port)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.sender.send_object(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    producer = FrameProducer('Lane Detection Test Video 01.mp4')
    producer.run()
