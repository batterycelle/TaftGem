import cv2
import argparse
import time
import os
from ultralytics import YOLO

output_folder = "[Insert Folder Here]"      # Folder where the images will be stored
VEHICLE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]

def main(video_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    sec_interval = 2
    img = 0
    last_saved_time = time.time()
    #variable declarations

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()

        current_time = time.time()

        if not ret:
            print("End of video or error reading frame")
            break  # End of video

        # Run YOLOv8 inference on the frame
        results = model(frame)  # results is a list of Results objects

        # Process the results
        for result in results:
            boxes = result.boxes  # Boxes object containing detections

            for box in boxes:
                cls = int(box.cls[0])  # Class index
                confidence = box.conf[0]  # Confidence score

                # Check if the detected class is a vehicle
                if cls in VEHICLE_CLASSES and confidence > 0.5:  # Threshold for confidence
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
                    label = f"Vehicle: {confidence:.2f}"

                    #Variable Declarations
                    current_time = time.time()
                    last_capture_time = last_saved_time

                    #Video segmentation process with minimum second intervals when vehicles are detected
                    if current_time - last_capture_time >= sec_interval:
                        last_capture_time = time.time()

                        #Filename and File Path Declarations
                        filename = os.path.join(output_folder, f'image_{img:03d}.png')  # Using f-string for formatting
                        img += 1

                        #Recording the Frame with Detected Vehicles
                        cv2.imwrite(filename, frame)
                        last_saved_time = last_capture_time

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Add label text
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the video with detections
        cv2.imshow('Vehicle Detection with YOLOv8', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Detect vehicles in a video using YOLOv8.')
    args.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parsed_args = args.parse_args()

    main(parsed_args.video)
