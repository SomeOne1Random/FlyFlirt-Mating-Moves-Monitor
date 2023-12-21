import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QLineEdit, QListWidget, QVBoxLayout,
                             QHBoxLayout, QTableWidget, QMessageBox, QGroupBox,  QScrollArea, QVBoxLayout, QWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import pandas as pd
from scipy import stats
'''
This is a PyQt6-based graphical user interface (GUI) application made for the analysis of fly behavior in video files. 
It incorporates various essential libraries including PyQt6 for the GUI components, OpenCV (`cv2`) for video processing,
Along with NumPy, Pandas, and SciPy for numerical operations, data handling, and statistical functions respectively.

The `VideoProcessingThread` class is dedicated to processing video files in a separate thread. 
This approach ensures efficient handling of tasks like reading video frames, detecting flies, tracking mating events. 
It also dynamically updates the GUI with critical data such as frame processing results and mating durations.

The user interface is run by the `MainWindow` class. It orchestrates the layout and functionality of the application.
It offers features like video display, control buttons for starting and stopping the analysis, and video navigation.
It also has dedicated sections for displaying mating information and analytical results.

Key functionalities of the program include the ability to select and process multiple video files.
This is with real-time display of frames and data extraction. The application analyzes for mating events, 
identifying these occurrences by monitoring the presence and count of flies within specified regions of interest (ROIs). 
Users are also provided with tools for managing these ROIs, including the option to void those deemed irrelevant.
Additionally, the application offers a data management and export feature.
This allows users to export the analyzed data such as mating times and durations into a CSV file for further analysis.
'''
class VideoProcessingThread(QThread):
    finished = pyqtSignal()
    frame_processed = pyqtSignal(str, np.ndarray, dict)
    frame_info = pyqtSignal(int, float)
    verified_mating_start_times = pyqtSignal(str, dict)
    void_roi_signal = pyqtSignal(str, int)  # Signal to emit with video_path and void ROI ID

    def __init__(self, video_path, initial_contours, fps, skip_frames=0, perf_frame_skips=1):
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
        self.latest_frames = {}  # Stores the latest frame for each video
        self.latest_mating_durations = {}  # Stores the latest mating durations for each video
        self.flies_count_signal = pyqtSignal(str, int,
                                             int)  # Signal to be emitted with video_path, ROI ID, and flies count
        self.flies_count_per_ROI = {}  # Tracks the count of flies per ROI
        self.void_rois = {}  # Dictionary to store void ROIs
        self.skip_frames = skip_frames  # Number of frames to skip from the beginning
        self.previous_flies_count_per_ROI = {}  # Tracks previous frame's fly count per ROI
        self.mating_event_detected = {}  # Tracks if a mating event is detected in an ROI
        self.previous_fly_positions_per_ROI = {}  # Tracks previous frame's fly positions per ROI when there are two flies
        self.mating_status_per_ROI = {}  # New dictionary to store mating status for each ROI
        self.mating_event_ongoing = {}  # Tracks ongoing mating events for each ROI
        self.perf_frame_skips = perf_frame_skips

    def export_combined_mating_times(self):
        # Initialize a dictionary to store combined mating times.
        combined_mating_times = {}

        # Iterate over each ROI and its corresponding mating start time.
        for roi_id, mating_time in self.mating_start_times.items():
            # Flag to check if current mating time is combined with an existing one.
            is_combined = False

            # Compare current mating time with already combined times.
            for combined_id, combined_time in combined_mating_times.items():
                # If the current mating time is within 1 second of an existing combined time,
                # combine these times by averaging them.
                if abs(mating_time - combined_time) <= 1:
                    combined_mating_times[combined_id] = (combined_time + mating_time) / 2
                    is_combined = True
                    break

            # If the current mating time was not combined with an existing one, add it as a new entry.
            if not is_combined:
                combined_mating_times[roi_id] = mating_time

        # Create a pandas DataFrame from the combined mating times.
        # The DataFrame contains two columns: 'ROI' and 'Start Time'.
        combined_mating_df = pd.DataFrame(list(combined_mating_times.items()), columns=['ROI', 'Start Time'])

        # Add a new column to the DataFrame for mating durations.
        # The mating duration for each ROI is fetched from the self.mating_durations dictionary.
        combined_mating_df['Mating Duration'] = [self.mating_durations.get(roi_id, 0) for roi_id in
                                                 combined_mating_df['ROI']]

        # Return the DataFrame containing the combined mating times and durations.
        return combined_mating_df

    def run(self):
        # Set the thread's running flag to True, indicating the thread is active.
        self.is_running = True

        # Clear the flies count data for each ROI when starting a new video analysis.
        self.flies_count_per_ROI.clear()

        # Open the video file for processing.
        cap = cv2.VideoCapture(self.video_path)

        # Skip a specified number of initial frames, if required.
        for _ in range(self.skip_frames):
            ret, _ = cap.read()
            if not ret:
                # If frame read fails, break from the loop.
                break

        # Initialize frame counters.
        frame_count = 0  # Counts every frame.
        current_frame = 0  # Counts frames considering skips for performance.

        # Main loop for video processing.
        while self.is_running:
            # Read a frame from the video.
            ret, frame = cap.read()
            if not ret:
                # If frame read fails or video ends, break from the loop.
                break

            # Process frames at intervals defined by perf_frame_skips for performance optimization.
            if current_frame % self.perf_frame_skips == 0:
                # Process the current frame to detect flies and analyze mating behavior.
                # The function process_frame can be customized based on specific detection algorithms.
                processed_frame, masks = self.process_frame(frame, self.initial_contours, frame_count)

                # Detect flies within the processed frame.
                # The detection algorithm can be adjusted according to the specifics of the video and target objects.
                self.detect_flies(processed_frame, masks, frame_count)

                # Emit the processed frame data along with current video path and mating durations.
                # This information is used to update the GUI.
                self.frame_processed.emit(self.video_path, processed_frame, self.mating_durations)

                # Update mating status for each ROI.
                for roi_id, is_mating in self.mating_event_ongoing.items():
                    self.mating_status_per_ROI[roi_id] = is_mating

                # Emit current frame information.
                # This includes the frame index and corresponding timestamp in the video.
                self.frame_info.emit(current_frame, current_frame / self.fps)

            # Increment frame counters.
            current_frame += 1
            frame_count += 1

        # Release the video capture object and free resources.
        cap.release()

        # Emit a signal to indicate that video processing is finished.
        self.finished.emit()

    def stop(self):
        self.is_running = False

    def process_frame(self, frame, initial_contours, frame_count):
        # Padding is added to the frame to handle objects at the edges more effectively.
        # These values can be adjusted based on the expected size and position of the objects in the frame.
        top_padding, bottom_padding, left_padding, right_padding = 50, 50, 50, 50

        # Add black padding to the frame
        frame_with_padding = cv2.copyMakeBorder(frame, top_padding, bottom_padding, left_padding, right_padding,
                                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Convert frame to grayscale for simplifying the processing and reducing computational complexity.
        gray = cv2.cvtColor(frame_with_padding, cv2.COLOR_BGR2GRAY)

        # Binary thresholding is applied for separating the objects (flies) from the background.
        # The threshold value (127) is chosen based on standard binary thresholding practices.
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Contours are detected to identify distinct objects (flies).
        # The RETR_EXTERNAL mode is used to retrieve only extreme outer contours, which is suitable for counting distinct objects.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Define a custom sorting function
        def custom_sort(contour):
            x, y, w, h = cv2.boundingRect(contour)
            y_tolerance = 200  # Adjust this tolerance as needed
            return (y // y_tolerance) * 1000 + x  # Sort primarily by y (with tolerance), and then by x

        # Convert contours to a list and sort based on x-coordinate of their bounding rectangles
        contours_list = list(contours)

        # Sort the contours using the custom sorting function
        contours_list.sort(key=custom_sort)

        # Store initial contours if frame_count
        if frame_count <= 1:
            initial_contours.clear()
            self.roi_ids.clear()  # Clear ROI IDs
            for i, contour in enumerate(contours_list):
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold to filter out noise
                    initial_contours.append({"contour": contour, "edge_duration": 0})
                    contour_id = self.generate_contour_id(contour)
                    self.roi_ids[contour_id] = i + 1  # Assign ID to ROI
        else:
            # Check for contours near the edges
            for contour_data in initial_contours:
                contour = contour_data["contour"]
                (x, y, w, h) = cv2.boundingRect(contour)
                if x <= 5 or y <= 5 or (x + w) >= frame_with_padding.shape[1] - 5 or (y + h) >= \
                        frame_with_padding.shape[0] - 5:
                    contour_data["edge_duration"] += 1
                else:
                    contour_data["edge_duration"] = 0

        # Calculate and round radii to find the mode radius
        radii = []
        for contour_data in initial_contours:
            (x, y, w, h) = cv2.boundingRect(contour_data["contour"])
            radii.append(int(round((w + h) / 4)))

        mode_radius = int(stats.mode(radii)[0]) if radii else 0  # Default to 0 if radii list is empty

        # Create masks and draw green circles using mode radius
        masks = []
        processed_frame = frame_with_padding.copy()
        for contour_data in initial_contours:
            (x, y, w, h) = cv2.boundingRect(contour_data["contour"])
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # Create mask with circle
            mask = np.zeros(processed_frame.shape[:2], dtype="uint8")
            cv2.circle(mask, (center_x, center_y), mode_radius, 255, -1)
            masks.append(mask)

            # Draw circle on processed frame
            if (x > 5 and y > 5 and (x + w) < frame_with_padding.shape[1] - 5 and (y + h) < frame_with_padding.shape[
                0] - 5) or contour_data["edge_duration"] >= 90:
                cv2.circle(processed_frame, (center_x, center_y), mode_radius, (0, 255, 0), 4)

        # Draw ROI numbers
        for i, contour_data in enumerate(initial_contours):
            (x, y, w, h) = cv2.boundingRect(contour_data["contour"])
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)

            # Determine position for ROI number
            text_position = (center_x, center_y - 55)
            cv2.putText(processed_frame, str(i), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 2,
                        cv2.LINE_AA)

        return processed_frame, masks

    def detect_flies(self, frame_with_padding, masks, frame_count):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        # Blob detection parameters are set specifically for identifying small, round objects like flies.
        # The parameters can be tweaked based on the size and shape of the flies in the video.
        params.minArea = 10  # Minimum area for a detected object to be considered a fly.        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        # Create blob detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Define radius and thickness for drawing circles
        radius = 6  # Increase the radius for larger dots
        thickness = -1  # Set the thickness to a negative value for a hollow circle

        # The grace frame threshold is set to allow a short period of time for flies to separate before ending a mating event.
        # This value can be adjusted based on the behavior of the flies and the frame rate of the video.
        grace_frames_threshold = int(self.fps * 3 / self.perf_frame_skips)

        # Iterate through each mask and detect flies
        for i, mask in enumerate(masks):
            # If an ROI has been marked as void, continue to the next ROI
            if self.void_rois.get(i, False):
                continue

            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame_with_padding, frame_with_padding, mask=mask)

            # Convert the masked frame to grayscale (if not already done)
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            # Kernel for morphological operations is designed to close small holes within detected objects, improving detection accuracy.
            kernel = np.ones((5, 5), np.uint8)

            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # Continue with the existing blob detection
            keypoints = detector.detect(gray)

            # Detect flies count for the first 500 frames
            if frame_count < 500:
                flies_count = len(keypoints)
                current_positions = [keypoint.pt for keypoint in keypoints]

                # Check for mating event when there's a transition from two to one fly
                if flies_count == 1 and i in self.previous_fly_positions_per_ROI:
                    prev_positions = self.previous_fly_positions_per_ROI[i]
                    # Ensure prev_positions has exactly two elements
                    if len(prev_positions) == 2:
                        distance_between_flies = np.linalg.norm(
                            np.array(prev_positions[0]) - np.array(prev_positions[1]))
                        if distance_between_flies > 30:
                            self.mating_event_detected[i] = True
                    # Clear the stored positions after checking
                    del self.previous_fly_positions_per_ROI[i]

                # Only store positions when there are exactly two flies
                elif flies_count == 2:
                    self.previous_fly_positions_per_ROI[i] = current_positions

                if i not in self.flies_count_per_ROI:
                    self.flies_count_per_ROI[i] = []
                self.flies_count_per_ROI[i].append(flies_count)

                self.previous_fly_positions_per_ROI[i] = current_positions

                # Check the condition after 200 frames
                if len(self.flies_count_per_ROI[i]) == 200:
                    more_than_two_count = sum(count > 2 for count in self.flies_count_per_ROI[i])
                    less_than_two_count = sum(count < 2 for count in self.flies_count_per_ROI[i])

                    # Calculate 75% of 200 frames
                    threshold = 200 * 0.75

                    # Adjust the logic to not mark the ROI as void if a mating event is detected
                    if more_than_two_count > threshold or (
                            less_than_two_count > threshold and not self.mating_event_detected.get(i, False)):
                        self.void_rois[i] = True
                        self.void_roi_signal.emit(self.video_path, i)  # Emit the signal

                # Mating event detection and handling
            if len(keypoints) == 1:  # A mating event is occurring
                self.mating_event_ongoing[i] = True

                # Start timing the mating event if not already started
                if i not in self.mating_start_frames:
                    self.mating_start_frames[i] = frame_count

                # Reset grace frames counter for this ROI
                self.mating_grace_frames[i] = 0

                # Calculate the duration of the mating event in frames and convert to seconds
                mating_duration = (frame_count - self.mating_start_frames[i]) / self.fps
                self.mating_durations.setdefault(i, []).append(mating_duration)  # Store duration in list per ROI

                # If mating duration exceeds 60 seconds and this ROI doesn't have a verified mating start time yet
                if mating_duration >= 360 and i not in self.mating_start_times:
                    mating_time = frame_count / self.fps
                    self.mating_start_times[i] = mating_time
                    # Emit the verified mating start times
                    self.verified_mating_start_times.emit(self.video_path, self.mating_start_times)

            else:  # Mating event has potentially ended
                self.mating_event_ongoing[i] = False
                self.mating_grace_frames[i] = self.mating_grace_frames.get(i, 0) + 1

                # If grace frames counter exceeds threshold, consider the mating event to have ended
                if self.mating_grace_frames[i] > grace_frames_threshold:
                    if i in self.mating_start_frames:
                        del self.mating_start_frames[i]
                        del self.mating_grace_frames[i]

                # Draw dots on the frame for each detected fly (centroid)
            for keypoint in keypoints:
                x = int(keypoint.pt[0])
                y = int(keypoint.pt[1])

                # Determine the color based on mating status and grace frame status
                if self.mating_event_ongoing.get(i, False) or self.mating_grace_frames.get(i,
                                                                                           0) <= grace_frames_threshold:
                    # Mating event is ongoing or within grace frame period
                    mating_duration = self.mating_durations.get(i, [0])[-1]  # Get the latest duration
                    if mating_duration < 360:
                        color = (0, 255, 255)  # Yellow dot
                    else:
                        color = (255, 0, 0)  # Blue dot
                else:
                    # Mating event has potentially ended
                    color = (0, 0, 255)  # Red dot

                cv2.circle(frame_with_padding, (x, y), radius, color, thickness)


        self.frame_processed.emit(self.video_path, frame_with_padding, self.mating_durations)

    def generate_contour_id(self, contour):
        return cv2.contourArea(contour)

    def void_roi(self, roi_id):
        self.void_rois[roi_id] = True
        self.void_roi_signal.emit(self.video_path, roi_id)  # Emit the signal to indicate a void ROI


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window attributes
        self.setWindowTitle("Fly Behavior Analysis")
        self.setGeometry(200, 200, 1200, 1400)  # Adjust size as needed

        # Paths and initial setups
        self.video_path = None
        self.initial_contours = []
        self.video_paths = []  # List to store multiple video paths
        self.video_threads = {}  # Dictionary to store threads for each video path
        self.current_video_index = 0  # Index to keep track of the currently displayed video
        self.latest_frames = {}  # Stores the latest frame for each video
        self.latest_mating_durations = {}  # Stores the latest mating durations for each video
        self.mating_start_times_dfs = {}  # Dictionary to store mating start times for each video
        # Organize UI elements
        self.init_ui()

    def init_ui(self):
        # Video Display & Info Section
        video_display_group = QGroupBox("Video Display", self)
        video_display_group.setGeometry(10, 10, 870, 500)

        vbox = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setFixedSize(860, 440)
        vbox.addWidget(self.video_label)

        hbox = QHBoxLayout()
        self.frame_label = QLabel('Frame: 0')
        hbox.addWidget(self.frame_label)
        self.time_label = QLabel('Time (s): 0')
        hbox.addWidget(self.time_label)
        vbox.addLayout(hbox)

        video_display_group.setLayout(vbox)

        # Video Control Section
        video_control_group = QGroupBox("Video Controls", self)
        video_control_group.setGeometry(10, 520, 870, 110)

        vbox = QVBoxLayout()

        self.fps_input = QLineEdit()
        self.fps_input.setPlaceholderText("Enter Video FPS")
        vbox.addWidget(self.fps_input)

        hbox = QHBoxLayout()
        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)
        hbox.addWidget(self.select_button)

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        hbox.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        hbox.addWidget(self.stop_button)

        vbox.addLayout(hbox)
        video_control_group.setLayout(vbox)

        # Video List Section
        video_list_group = QGroupBox("Video List", self)
        video_list_group.setGeometry(10, 640, 870, 120)

        vbox = QVBoxLayout()

        self.video_list_widget = QListWidget()
        vbox.addWidget(self.video_list_widget)

        video_list_group.setLayout(vbox)

        # Mating Information Display Area
        mating_info_area = QWidget(self)
        mating_info_area.setGeometry(10, 750, 870, 150)

        hbox = QHBoxLayout()

        # Mating Duration Display with Scrollable Area
        mating_duration_group = QGroupBox("Mating Durations", mating_info_area)
        vbox = QVBoxLayout()
        self.mating_duration_label = QLabel("Mating Durations:")
        vbox.addWidget(self.mating_duration_label)

        # Create a scroll area for mating durations
        mating_duration_scroll = QScrollArea()
        mating_duration_scroll.setWidgetResizable(True)
        mating_duration_scroll.setWidget(mating_duration_group)
        mating_duration_group.setLayout(vbox)
        hbox.addWidget(mating_duration_scroll)

        # Verified Mating Times Display with Scrollable Area
        verified_times_group = QGroupBox("Verified Mating Times", mating_info_area)
        vbox = QVBoxLayout()
        self.verified_mating_times_label = QLabel("Verified Mating Times:")
        vbox.addWidget(self.verified_mating_times_label)

        # Create a scroll area for verified mating times
        verified_times_scroll = QScrollArea()
        verified_times_scroll.setWidgetResizable(True)
        verified_times_scroll.setWidget(verified_times_group)
        verified_times_group.setLayout(vbox)
        hbox.addWidget(verified_times_scroll)

        mating_info_area.setLayout(hbox)

        # Navigation Controls
        nav_group = QGroupBox("Navigation", self)
        nav_group.setGeometry(10, 900, 870, 80)

        hbox = QHBoxLayout()

        # Use arrow icons for previous and next buttons
        self.prev_button = QPushButton("← Previous Video")
        self.prev_button.clicked.connect(self.previous_video)
        hbox.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Video →")
        self.next_button.clicked.connect(self.next_video)
        hbox.addWidget(self.next_button)

        nav_group.setLayout(hbox)

        # Export Functionality
        export_group = QGroupBox("Data Export", self)
        export_group.setGeometry(10, 965, 870, 80)

        self.skip_frames_input = QLineEdit(self)
        self.skip_frames_input.setPlaceholderText("Enter number of seconds to skip")
        self.skip_frames_input.setGeometry(1000, 20, 200, 30)  # x, y, width, height

        hbox = QHBoxLayout()

        self.export_button = QPushButton("Export DataFrame")
        self.export_button.clicked.connect(self.export_dataframe)
        self.export_button.setToolTip("Export the mating data as a CSV file.")
        hbox.addWidget(self.export_button)

        self.processing_status_label = QLabel("Status: Awaiting action.")
        hbox.addWidget(self.processing_status_label)

        export_group.setLayout(hbox)


        self.roi_control_group = QGroupBox("Manual ROI Control", self)
        self.roi_control_group.setGeometry(890, 35, 300, 80)  # Adjust the position and size as needed

        roi_control_layout = QHBoxLayout()

        # Modify or replace the existing ROI control group
        self.roi_control_group = QGroupBox("Manual ROI Control", self)
        self.roi_control_group.setGeometry(890, 35, 300, 200)  # Adjust the position and size as needed

        roi_control_layout = QVBoxLayout()  # Changed to QVBoxLayout for better alignment

        # Add a QLineEdit for multiple ROI IDs
        self.multi_roi_input = QLineEdit(self)
        self.multi_roi_input.setPlaceholderText("Enter multiple ROI IDs separated by commas")
        roi_control_layout.addWidget(self.multi_roi_input)

        # Add a button for voiding multiple ROIs
        self.void_multi_roi_button = QPushButton("Void Multiple ROIs", self)
        self.void_multi_roi_button.clicked.connect(self.void_multiple_rois)
        roi_control_layout.addWidget(self.void_multi_roi_button)

        # Add a QListWidget to display ROI voiding status
        self.roi_void_list = QListWidget(self)
        roi_control_layout.addWidget(self.roi_void_list)

        self.roi_control_group.setLayout(roi_control_layout)

        # Add an input for frame skip value
        self.frame_skip_input = QLineEdit(self)
        self.frame_skip_input.setPlaceholderText("Enter Frame Skip Value")

        # Set the geometry of the frame skip input (x, y, width, height)
        self.frame_skip_input.setGeometry(890, 230, 160, 30)  # Adjust these values as needed

    # Handle errors or other information that needs to be shown to the user
    def show_error(self, message):
        # Display a critical error message to the user.
        # This method is used throughout the application to inform the user of errors in a standard format.
        QMessageBox.critical(self, "Error", message)

    def show_info(self, title, message):
        # Display an informational message to the user.
        # This method standardizes the display of non-critical information, such as confirmations or general notices.
        QMessageBox.information(self, title, message)

    def void_roi(self):
        # Function to mark a Region of Interest (ROI) as void (i.e., to be ignored in further processing).
        try:
            # Convert the text input for ROI ID to an integer.
            # This assumes the user inputs a valid integer ID for the ROI.
            roi_id = int(self.roi_id_input.text())

            # Retrieve the path of the currently selected video.
            current_video_path = self.video_paths[self.current_video_index]

            # Fetch the corresponding video processing thread for the current video.
            # Each video has a dedicated thread for processing.
            video_thread = self.video_threads.get(current_video_path)

            if video_thread:
                # If the video thread exists, void the specified ROI in that video.
                video_thread.void_roi(roi_id)

                # Log the action to the console for debugging or auditing purposes.
                print(f"Manually voided ROI {roi_id} in video {current_video_path}")
            else:
                # If the video thread does not exist, display an error message to the user.
                self.show_error("No video thread found for the current video.")
        except ValueError:
            # If the ROI ID input is not a valid integer, display an error message to the user.
            self.show_error("Invalid ROI ID entered.")

    def void_multiple_rois(self):
        # Function to void multiple Regions of Interest (ROIs) based on user input.
        # This allows the user to ignore multiple ROIs in the analysis process efficiently.

        # Retrieve and clean the text input for ROI IDs.
        # The input is expected to contain ROI IDs separated by commas, possibly with ranges indicated by dashes.
        roi_input = self.multi_roi_input.text().strip()
        roi_ids = []

        # Split the input text into individual ROI entries.
        roi_entries = roi_input.split(',')

        for entry in roi_entries:
            # Clean each entry to remove extra whitespace.
            entry = entry.strip()

            # If an entry contains a dash, it indicates a range of ROIs.
            if '-' in entry:
                # Split the range and convert the start and end values to integers.
                start, end = map(int, entry.split('-'))

                # Add all ROI IDs within the specified range to the list.
                roi_ids.extend(range(start, end + 1))
            else:
                # If the entry is a single number, try converting it to an integer and add it to the list.
                try:
                    roi_id = int(entry)
                    roi_ids.append(roi_id)
                except ValueError:
                    # If the conversion fails, show an error message indicating the invalid entry.
                    self.show_error(f"Invalid ROI ID or range: {entry}")

        # Retrieve the current video path from the list of video paths.
        current_video_path = self.video_paths[self.current_video_index]

        # Get the corresponding video processing thread.
        video_thread = self.video_threads.get(current_video_path)

        # If the video thread exists, proceed to void the specified ROIs.
        if video_thread:
            for roi_id in roi_ids:
                # Void each ROI and update the ROI void list widget.
                # This also logs the action for auditing purposes.
                video_thread.void_roi(roi_id)
                self.roi_void_list.addItem(f"ROI {roi_id} voided in video {current_video_path}")
                print(f"Manually voided ROI {roi_id} in video {current_video_path}")
        else:
            # If no video thread is found, display an error message.
            self.show_error("No video thread found for the current video.")

    def add_export_button(self):
        # Create and set up the 'Export DataFrame' button.
        # This button is used to export the analyzed data into a CSV file.

        self.export_button = QPushButton("Export DataFrame", self)

        # Set the geometry (position and size) of the button on the GUI.
        self.export_button.setGeometry(10, 480, 780, 30)

        # Connect the button's click event to the function that handles data export.
        self.export_button.clicked.connect(self.export_dataframe)

        # Initially, the button is disabled and will be enabled when data is ready for export.
        self.export_button.setEnabled(False)

    def enable_export_button(self):
        # Enable the 'Export DataFrame' button.
        # This function is called when the data is ready to be exported.
        self.export_button.setEnabled(True)

    def export_dataframe(self):
        # Function to export the analyzed mating data to a CSV file for each video.
        for video_path, video_thread in self.video_threads.items():
            # Check if the video thread exists and has processed data.
            if video_thread:
                # Generate a default export filename based on the original video filename.
                default_export_name = video_path.rsplit('/', 1)[-1].rsplit('.', 1)[0] + '.csv'

                # Open a file dialog for the user to choose the save location and file name.
                file_path, _ = QFileDialog.getSaveFileName(self, f"Export DataFrame for {video_path}",
                                                           default_export_name, "CSV Files (*.csv);;All Files (*)")

                # Proceed with the export if a file path is provided.
                if file_path:
                    # Prepare the data for export by iterating over each ROI.
                    data = []
                    num_rois = len(video_thread.initial_contours)
                    for roi in range(num_rois):
                        # Calculate and adjust the mating start times and durations.
                        start_time = video_thread.mating_start_times.get(roi, 'N/A')
                        start_time = 'N/A' if start_time == 'N/A' else max(0, start_time - 360)

                        durations = video_thread.mating_durations.get(roi, [])
                        longest_duration = max(durations, default=0)
                        longest_duration = 0 if longest_duration < 360 else longest_duration

                        # Determine the mating status based on the duration of the most recent mating event.
                        mating_status = durations[-1] >= 360 if durations else False

                        # Append the calculated data for each ROI to the data list.
                        data.append({'ROI': roi, 'Adjusted Start Time': start_time,
                                     'Longest Duration': longest_duration, 'Mating Status': mating_status})

                    # Create a DataFrame from the prepared data.
                    mating_times_df = pd.DataFrame(data)

                    # Mark ROIs that have been voided as 'N/A' in the DataFrame.
                    void_rois = video_thread.void_rois
                    for column in ['Adjusted Start Time', 'Longest Duration', 'Mating Status']:
                        mating_times_df[column] = mating_times_df.apply(
                            lambda row: 'N/A' if void_rois.get(row['ROI'], False) else row[column], axis=1)

                    # Export the DataFrame to a CSV file.
                    mating_times_df.to_csv(file_path, index=False)
                    self.processing_status_label.setText('DataFrame exported successfully.')
                    QMessageBox.information(self, "Success", "DataFrame exported successfully.")
                else:
                    # Display a warning if the export is canceled or fails.
                    QMessageBox.warning(self, "Warning", "Export was canceled or failed.")

    def previous_video(self):
        # Function to navigate to the previous video in the video list.
        if self.current_video_index > 0:
            # Decrease the current video index to move to the previous video.
            self.current_video_index -= 1

            # Retrieve the path of the previous video.
            current_video_path = self.video_paths[self.current_video_index]

            # Update the video frame display with the frame data of the previous video.
            if current_video_path in self.latest_frames:
                frame = self.latest_frames[current_video_path]
                mating_durations = self.latest_mating_durations[current_video_path]
                self.update_video_frame(current_video_path, frame, mating_durations)

    def next_video(self):
        # Function to navigate to the next video in the video list.
        if self.current_video_index < len(self.video_paths) - 1:
            # Increase the current video index to move to the next video.
            self.current_video_index += 1

            # Retrieve the path of the next video.
            current_video_path = self.video_paths[self.current_video_index]

            # Update the video frame display with the frame data of the next video.
            if current_video_path in self.latest_frames:
                frame = self.latest_frames[current_video_path]
                mating_durations = self.latest_mating_durations[current_video_path]
                self.update_video_frame(current_video_path, frame, mating_durations)

    def update_verified_mating_times(self, video_path, mating_times_dict):
        # Function to update the GUI with verified mating times for each ROI.
        # It checks if the mating times are available for the given video path.

        if video_path in self.video_threads and hasattr(self.video_threads[video_path], 'mating_durations'):
            # Adjust mating start times by subtracting a fixed duration (360 seconds) to account for any predefined offsets.
            # The offset (360 seconds) is chosen based on specific criteria or observations made during the video analysis.
            adjusted_mating_times_dict = {roi_id: max(0, time - 360) for roi_id, time in mating_times_dict.items()}

            # Create a DataFrame for easier handling and display of mating times.
            mating_times_df = pd.DataFrame(list(adjusted_mating_times_dict.items()), columns=['ROI', 'Start Time'])

            # Retrieve mating durations for each ROI from the current video thread.
            durations = self.video_threads[video_path].mating_durations

            # Add mating durations to the DataFrame.
            mating_times_df['Mating Duration'] = [durations.get(roi_id, 0) for roi_id in mating_times_df['ROI']]

            # Store the DataFrame in a dictionary for future reference or export.
            self.mating_start_times_dfs[video_path] = mating_times_df

            # Check if the current video being displayed matches the video path of the updated data.
            current_video_path = self.video_paths[self.current_video_index]
            if video_path == current_video_path:
                # Enable the export button as the data is now available for export.
                self.enable_export_button()

                # Update the GUI label to display the verified mating times.
                mating_time_text = "\n".join(
                    [f"ROI {roi_id}: {time:.2f} seconds" for roi_id, time in adjusted_mating_times_dict.items()])
                self.verified_mating_times_label.setText(mating_time_text)
        else:
            # Log an error if the mating durations attribute is missing in the video thread.
            print("Error: video_thread for the current video is None or doesn't have the required attributes")

    def set_fps_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.fps_input.setText(str(fps))

    def select_video(self):
        # Function to select one or more video files using a file dialog.
        # It allows batch processing of multiple videos.

        # Open a file dialog to choose video files. Multiple file selection is enabled.
        video_paths, _ = QFileDialog.getOpenFileNames(self, "Select Videos")

        # Check if any video paths were selected.
        if video_paths:
            # Extend the existing list of video paths with the newly selected paths.
            self.video_paths.extend(video_paths)

            # For each selected video path:
            for video_path in video_paths:
                # Set the frames per second (fps) for the video.
                self.set_fps_from_video(video_path)

                # Initialize the corresponding thread for each video as None.
                self.video_threads[video_path] = None

                # Add the video filename to the GUI list widget for display.
                self.video_list_widget.addItem(video_path.split("/")[-1])

            # Enable the 'Start Processing' button only if at least one video is selected.
            self.start_button.setEnabled(len(self.video_paths) > 0)

    def start_processing(self):
        # Function to start processing the selected videos.
        # It checks if there are video paths available and if the fps input is set.

        if self.video_paths and self.fps_input.text():
            # Disable the 'Start Processing' and 'Select Video' buttons to prevent re-triggering during processing.
            self.start_button.setEnabled(False)
            self.select_button.setEnabled(False)

            # Enable the 'Stop Processing' button to allow stopping the process if needed.
            self.stop_button.setEnabled(True)

            # Retrieve and set the fps value from the user input.
            fps = float(self.fps_input.text())

            # Calculate the number of frames to skip at the start based on the input seconds.
            skip_seconds = float(self.skip_frames_input.text()) if self.skip_frames_input.text() else 0
            skip_frames = int(skip_seconds * fps)  # Convert the skip time from seconds to frames.

            # Set up frame skips for performance optimization, defaulting to 1 if input is invalid.
            try:
                perf_frame_skips = int(self.frame_skip_input.text())
            except ValueError:
                perf_frame_skips = 1

            # For each video path, create and start a video processing thread.
            for video_path in self.video_paths:
                # Check if the thread for this path is not already created or started.
                if video_path not in self.video_threads or not self.video_threads[video_path]:
                    # Create a new VideoProcessingThread object.
                    video_thread = VideoProcessingThread(video_path, [], fps, skip_frames, perf_frame_skips)

                    # Store the thread object in the dictionary.
                    self.video_threads[video_path] = video_thread

                    # Connect various signals from the thread to the corresponding slots in the GUI.
                    video_thread.verified_mating_start_times.connect(self.update_verified_mating_times)
                    video_thread.frame_info.connect(self.update_frame_info)
                    video_thread.frame_processed.connect(self.update_video_frame)
                    video_thread.finished.connect(self.processing_finished)
                    video_thread.void_roi_signal.connect(self.void_roi_handler)

                    # Start the video processing thread.
                    video_thread.start()

            # Enable navigation buttons (Previous, Next) if there are multiple videos.
            self.prev_button.setEnabled(len(self.video_paths) > 1)
            self.next_button.setEnabled(len(self.video_paths) > 1)

    def stop_processing(self):
        # Stop all video threads
        for video_thread in self.video_threads.values():
            if video_thread and video_thread.is_running:
                video_thread.stop()
        # Update the status label to indicate that processing has stopped
        self.processing_status_label.setText('Video processing stopped.')
        # Re-enable the start and select buttons
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def processing_finished(self):
        self.processing_status_label.setText('Video processing finished.')
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_video_frame(self, video_path, frame, mating_durations):
        # Function to update the video display and mating information in the GUI.

        # Check if the frame is from the currently selected video.
        current_video_path = self.video_paths[self.current_video_index]
        if video_path == current_video_path:
            # Update the display with the current frame.

            # Extract frame dimensions and convert the frame to a QImage for display in the GUI.
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

            # Convert QImage to QPixmap and scale it to fit the video_label widget.
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(),
                                   Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(pixmap)

            # Update the mating_duration_label with the latest mating duration for each ROI.
            mating_duration_text = ""
            for roi_id, durations in mating_durations.items():
                # Get the most recent mating duration for each ROI, default to 0 if no data.
                latest_duration = durations[-1] if durations else 0
                mating_duration_text += f"ROI {roi_id}: {latest_duration:.2f} seconds\n"
            self.mating_duration_label.setText(mating_duration_text)

            # If mating start times are available for the current video, display them.
            if video_path in self.mating_start_times_dfs:
                mating_times_df = self.mating_start_times_dfs[video_path]
                mating_time_text = "\n".join(
                    [f"ROI {row['ROI']}: {row['Start Time']:.2f} seconds" for _, row in mating_times_df.iterrows()])
                self.verified_mating_times_label.setText(mating_time_text)
            else:
                # Clear the label if no data is available.
                self.verified_mating_times_label.setText("")

        # Store the latest frame and mating durations for the current video.
        # This is useful for switching between videos without reprocessing frames.
        self.latest_frames[video_path] = frame
        self.latest_mating_durations[video_path] = mating_durations

    def update_frame_info(self, frame, time):
        self.frame_label.setText(f'Frame: {frame}')
        self.time_label.setText(f'Time (s): {time:.2f}')

    def void_roi_handler(self, video_path, roi_id):
        # Handle the void ROI, perhaps by updating the UI or logging
        print(f"ROI {roi_id} in video {video_path} has been marked as void.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())