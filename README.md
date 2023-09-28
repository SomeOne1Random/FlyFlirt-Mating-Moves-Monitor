# Drosophila melanogaster Latency Copulation Duration Finder
To provide clear instructions on the homepage of your GitHub repository, you can create a `README.md` file. This file is automatically displayed on the homepage of your repository and is a great place to explain what your project does and how to install and use it. Here's a template you can follow:

This is a Python application for analyzing fly behavior in videos. It uses PyQt6 for the GUI and OpenCV for video processing.

## Features

- Analyze fly behavior in videos.
- Export mating start times and durations to a CSV file.
- Real-time display of video frames with detected flies.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fly-behavior-analysis.git
   ```

2. Navigate to the project directory:

   ```bash
   cd fly-behavior-analysis
   ```

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:

   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

5. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   python main.py
   ```

2. Select a video file using the "Select Video" button.
3. Enter the video's frames per second (FPS) in the input field.
4. Click "Start Processing" to begin analyzing the video.
5. The processed video frames will be displayed in real-time, and mating durations will be shown in the GUI.
6. You can export the mating start times and durations to a CSV file using the "Export DataFrame" button.

## Contributing

Contributions are welcome! If you find any issues or have improvements to suggest, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Replace `"https://github.com/yourusername/fly-behavior-analysis.git"` with the actual URL of your GitHub repository.

Make sure to provide clear and concise instructions for installation, usage, and contributing to your project. Including a license is also essential for making your project open source.

Once you've created the `README.md` file, it will be displayed on your GitHub repository's homepage, providing a user-friendly introduction to your project.
