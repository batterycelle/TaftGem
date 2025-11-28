pip install opencv-contrib-python

import cv2

def capture_frames_at_interval(video_path, interval_seconds, output_folder):
    """
    Captures frames from a video at specified intervals and saves them as images.

    Args:
    video_path (str): Path to the video file.
    interval_seconds (int): Time interval between frames in seconds.
    output_folder (str): Path to the folder where frames will be saved.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Fixed the f-string syntax and used raw string (r prefix) for Windows paths
            output_path = f"{output_folder}/frame{saved_count:04d}.jpg"
            cv2.imwrite(output_path, frame)
            print(f"Saved {output_path}")
            saved_count += 1
        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()

# Use raw strings (r prefix) for Windows paths to avoid escape character issues
video_path = r"D:\GatoCode\FloodLevel1.mp4"
interval_seconds = 1  # Capture a frame every 1 seconds

# Specify the full path for the output folder
# For example, to save in D:\GatoDrone\frames
output_folder = r"D:\GatoDrone\frames"  # Use your desired directory path here

capture_frames_at_interval(video_path, interval_seconds, output_folder)

