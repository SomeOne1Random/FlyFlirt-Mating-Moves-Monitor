import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QLineEdit
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import pandas as pd

class VideoProcessingThread(QThread):
    finished = pyqtSignal()
    frame_processed = pyqtSignal(np.ndarray, dict)
    frame_info = pyqtSignal(int, float)
    verified_mating_start_times = pyqtSignal(dict)

    def __init__(self, video_path, initial_contours, fps):
        super().__init__()
        self.video_path = video_path
        self.initial_contours = initial_contours
        self.is_running = False
        self.roi_ids = {}  # Dictionary to store ROI IDs
        self.mating_start_times = {}  # Dictionary to store mating start times for each ROI
        self.mating_durations = {}  # Dictionary to store mating durations for each ROI
        self.fps = fps
        self.mating_start_frames = {}  # Dictionary to store mating start frames for each ROI
        self.mating_grace_frames = {}  # Dictionary to store grace frames for each ROI
        self.mating_start_times_df = pd.DataFrame(columns=['ROI', 'Start Times'])  # Create an empty DataFrame to store mating start times

    def export_combined_mating_times(self):
        combined_mating_times = {}

        for roi_id, mating_time in self.mating_start_times.items():
            # Check if this mating time is within 1 second of another mating time
            is_combined = False
            for combined_id, combined_time in combined_mating_times.items():
                if abs(mating_time - combined_time) <= 1:
                    combined_mating_times[combined_id] = (combined_time + mating_time) / 2
                    is_combined = True
                    break

            if not is_combined:
                combined_mating_times[roi_id] = mating_time

        # Create a DataFrame from the combined mating times
        combined_mating_df = pd.DataFrame(list(combined_mating_times.items()), columns=['ROI', 'Start Time'])
        combined_mating_df['Mating Duration'] = [self.mating_durations.get(roi_id, 0) for roi_id in
                                                 combined_mating_df['ROI']]

        return combined_mating_df


    def run(self):
        self.is_running = True

        cap = cv2.VideoCapture(self.video_path)

        frame_count = 0
        current_frame = 0
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame, self.initial_contours, current_frame)  # or any other value for frame_count
            self.frame_info.emit(current_frame, current_frame / self.fps)
            current_frame += 1

            # Process the frame here
            processed_frame, masks = self.process_frame(frame, self.initial_contours, frame_count)
            self.detect_flies(processed_frame, masks, frame_count)

            self.frame_processed.emit(processed_frame, self.mating_durations)

            frame_count += 1

        cap.release()
        self.finished.emit()

    def stop(self):
        self.is_running = False

    def process_frame(self, frame, initial_contours, frame_count):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold the image to obtain a binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours of white regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to a list and sort based on x-coordinate of their bounding rectangles
        contours_list = list(contours)
        contours_list.sort(key=lambda ctr: cv2.boundingRect(ctr)[0])

        # Store initial contours if frame_count <= 100
        if frame_count <= 100:
            initial_contours.clear()
            self.roi_ids.clear()  # Clear ROI IDs
            for i, contour in enumerate(contours_list):
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold to filter out noise
                    initial_contours.append({"contour": contour, "edge_duration": 0})
                    contour_id = self.generate_contour_id(contour)
                    self.roi_ids[contour_id] = i + 1  # Assign ID to ROI
        else:
            # Check for contours near the edges
            for contour_data in initial_contours:
                contour = contour_data["contour"]
                (x, y, w, h) = cv2.boundingRect(contour)
                if x <= 5 or y <= 5 or (x + w) >= frame.shape[1] - 5 or (y + h) >= frame.shape[0] - 5:
                    contour_data["edge_duration"] += 1
                else:
                    contour_data["edge_duration"] = 0

        # Create masks based on the initial contours
        masks = []
        for contour_data in initial_contours:
            mask = np.zeros_like(gray)
            ellipse = cv2.fitEllipse(contour_data["contour"])
            cv2.ellipse(mask, ellipse, 255, -1)
            masks.append(mask)

        # Draw green circles and highlight areas where circles touch the black background
        processed_frame = frame.copy()
        for contour_data in initial_contours:
            contour = contour_data["contour"]
            edge_duration = contour_data["edge_duration"]
            (x, y, w, h) = cv2.boundingRect(contour)
            # Exclude contours near the edges of the frame if edge duration is less than a threshold
            if (x > 5 and y > 5 and (x + w) < frame.shape[1] - 5 and (y + h) < frame.shape[
                0] - 5) or edge_duration >= 90:
                cv2.circle(processed_frame, (int(x + w / 2), int(y + h / 2)), int((w + h) / 4), (0, 255, 0), 2)
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(processed_frame, ellipse, (0, 255, 0), 2)

        return processed_frame, masks

    def detect_flies(self, frame, masks, frame_count):
        # Initialize blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 10
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        # Create blob detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Define radius and thickness for drawing circles
        radius = 6  # Increase the radius for larger dots
        thickness = -1  # Set the thickness to a negative value for a hollow circle

        grace_frames_threshold = int(self.fps)  # Number of frames equivalent to 1 second

        # Iterate through each mask and detect flies
        for i, mask in enumerate(masks):
            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert the masked frame to grayscale
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            # Detect blobs (flies)
            keypoints = detector.detect(gray)

            # Draw dots on the frame for each detected fly (centroid), color depends on mating status
            if len(keypoints) == 1:  # A mating event is occurring
                x = int(keypoints[0].pt[0])
                y = int(keypoints[0].pt[1])

                # Start timing the mating event if not already started
                if i not in self.mating_start_frames:
                    self.mating_start_frames[i] = frame_count

                # Reset grace frames counter for this ROI
                self.mating_grace_frames[i] = 0

                # Calculate the duration of the mating event in frames and convert to seconds
                mating_duration = (frame_count - self.mating_start_frames[i]) / self.fps
                self.mating_durations.setdefault(i, []).append(mating_duration)  # Store duration in list per ROI

                # Change dot color based on mating duration
                if mating_duration < 360:  # Less than 1 minute
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), thickness)  # Yellow dot
                else:  # Over 1 minute
                    cv2.circle(frame, (x, y), radius, (255, 0, 0), thickness)  # Blue dot

                    # If mating duration exceeds 60 seconds and this ROI doesn't have a verified mating start time yet
                    if i not in self.mating_start_times:
                        mating_time = frame_count / self.fps
                        self.mating_start_times[i] = mating_time
                        # Emit the verified mating start times
                        self.verified_mating_start_times.emit(self.mating_start_times)

            else:  # Mating event has potentially ended
                # Increase grace frames counter for this ROI
                self.mating_grace_frames[i] = self.mating_grace_frames.get(i, 0) + 1

                # If grace frames counter exceeds threshold, consider the mating event to have ended
                if self.mating_grace_frames[i] > grace_frames_threshold:
                    if i in self.mating_start_frames:
                        del self.mating_start_frames[i]
                        del self.mating_grace_frames[i]

                for keypoint in keypoints:
                    x = int(keypoint.pt[0])
                    y = int(keypoint.pt[1])
                    cv2.circle(frame, (x, y), radius, (0, 0, 255), thickness)  # Red dot

        self.frame_processed.emit(frame, self.mating_durations)  # Emit the frame and the mating durations

    def generate_contour_id(self, contour):
        return cv2.contourArea(contour)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fly Behavior Analysis")
        self.setGeometry(200, 200, 800, 700)

        self.video_path = None
        self.initial_contours = []

        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 10, 780, 440)

        self.mating_duration_label = QLabel(self)
        self.mating_duration_label.setGeometry(10, 360, 700, 150)

        self.frame_label = QLabel('Frame: 0', self)
        self.frame_label.move(200, 410)
        self.time_label = QLabel('Time (s): 0', self)
        self.time_label.move(200, 430)
        self.verified_mating_times_label = QLabel(self)
        self.verified_mating_times_label.setGeometry(330, 350, 780, 120)


        self.fps_input = QLineEdit(self)
        self.fps_input.setGeometry(10, 630, 780, 30)
        self.fps_input.setPlaceholderText("Enter Video FPS")

        self.select_button = QPushButton("Select Video", self)
        self.select_button.setGeometry(10, 510, 780, 30)
        self.select_button.clicked.connect(self.select_video)


        self.start_button = QPushButton("Start Processing", self)
        self.start_button.setGeometry(10, 550, 780, 30)
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)  # The button is initially disabled

        self.stop_button = QPushButton("Stop Processing", self)
        self.processing_status_label = QLabel(self)
        self.processing_status_label.setGeometry(10, 670, 780, 30)
        self.stop_button.setGeometry(10, 590, 780, 30)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)  # The button is initially disabled

        self.video_thread = None
        self.add_export_button()  # Add this line to create the "Export DataFrame" button

    def add_export_button(self):
        self.export_button = QPushButton("Export DataFrame", self)
        self.export_button.setGeometry(10, 480, 780, 30)
        self.export_button.clicked.connect(self.export_dataframe)
        self.export_button.setEnabled(False)  # The button is initially disabled

    def enable_export_button(self):
        self.export_button.setEnabled(True)

    def export_dataframe(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export DataFrame to CSV", "",
                                                   "CSV Files (*.csv);;All Files (*)")
        if file_path:
            mating_start_times = self.video_thread.mating_start_times

            # Adjust mating start times by subtracting 360 seconds
            adjusted_mating_start_times = {roi_id: max(0, time - 360) for roi_id, time in mating_start_times.items()}

            mating_times_df = pd.DataFrame(adjusted_mating_start_times.items(), columns=['ROI', 'Start Time'])

            # Calculate longest mating duration for each ROI from the stored lists
            longest_mating_durations = {}
            for roi_id, durations in self.video_thread.mating_durations.items():
                longest_mating_durations[roi_id] = max(durations, default=0)  # Use max() with default for empty lists

            mating_times_df['Longest Duration'] = [longest_mating_durations.get(roi_id, 0) for roi_id in
                                                   mating_times_df['ROI']]

            mating_times_df.to_csv(file_path, index=False)
            self.processing_status_label.setText('DataFrame exported successfully.')

    def update_verified_mating_times(self, mating_times_dict):
        # Subtract 360 seconds from the verified mating start times
        adjusted_mating_times_dict = {roi_id: max(0, time - 360) for roi_id, time in mating_times_dict.items()}

        # Update verified mating times in the DataFrame
        self.mating_start_times_df = pd.DataFrame(list(adjusted_mating_times_dict.items()), columns=['ROI', 'Start Time'])
        self.mating_start_times_df['Mating Duration'] = [self.video_thread.mating_durations.get(roi_id, 0) for
                                                         roi_id in self.mating_start_times_df['ROI']]

        # Call the method to enable the "Export DataFrame" button
        self.enable_export_button()

        # Update verified mating times in the GUI
        mating_time_text = "\n".join(
            [f"ROI {roi_id}: {time:.2f} seconds" for roi_id, time in adjusted_mating_times_dict.items()])
        self.verified_mating_times_label.setText(mating_time_text)

    def set_fps_from_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.fps_input.setText(str(fps))

    def select_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Select Video")
        if self.video_path:
            self.set_fps_from_video()
        self.initial_contours.clear()
        # The start button is enabled only if the FPS input is not empty
        self.start_button.setEnabled(bool(self.fps_input.text()))

    def start_processing(self):
        if self.video_path and self.fps_input.text():
            self.start_button.setEnabled(False)
            self.select_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            fps = float(self.fps_input.text())
            self.video_thread = VideoProcessingThread(self.video_path, self.initial_contours, fps)
            self.video_thread.verified_mating_start_times.connect(self.update_verified_mating_times)
            self.video_thread.frame_info.connect(self.update_frame_info)
            self.video_thread.frame_processed.connect(self.update_video_frame)
            self.video_thread.finished.connect(self.processing_finished)
            self.video_thread.start()

    def stop_processing(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()

    def processing_finished(self):
        self.processing_status_label.setText('Video processing finished.')
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_video_frame(self, frame, mating_durations):
        # Update video label
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

        # Update mating duration label with the newest duration for each ROI
        mating_duration_text = ""
        for roi_id, durations in mating_durations.items():
            latest_duration = durations[-1] if durations else 0
            mating_duration_text += f"ROI {roi_id}: {latest_duration:.2f} seconds\n"

        self.mating_duration_label.setText(mating_duration_text)

    def update_frame_info(self, frame, time):
        self.frame_label.setText(f'Frame: {frame}')
        self.time_label.setText(f'Time (s): {time:.2f}')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())