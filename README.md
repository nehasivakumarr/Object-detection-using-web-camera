# YOLOv8 Object Detection Using Laptop Camera

## Aim
To access the laptop camera, capture an image, and detect objects using YOLOv8.

## Requirements
- Anaconda
- Jupyter Notebook
- Laptop Camera

## Steps
- Open Jupyter Notebook using Anaconda.
- Access your laptop camera using OpenCV.
- Wait for 5 seconds and capture an image.
- Display the captured image.
- Apply YOLOv8 object detection.
- Display the detected image with bounding boxes and labels

## Algorithm

```
Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects
```

## Program 

```
import cv2
from IPython.display import clear_output
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Variable to hold the last valid detected frame
last_frame = None

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Streaming started. Stop/Interrupt the cell when you want to freeze the final frame.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Object detection
            results = model(frame, conf=0.60, verbose=False)

            # Annotate and convert BGR to RGB
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Store the current frame as the last active frame
            last_frame = annotated_frame

            # Live update inside Jupyter
            clear_output(wait=True)
            plt.figure(figsize=(10, 8))
            plt.imshow(annotated_frame)
            plt.title("Real-Time Object Detection (Interrupt to Stop)")
            plt.axis("off")
            plt.show()

    except KeyboardInterrupt:
        print("\nStream stopped by user.")
    finally:
        # Release the webcam hardware
        cap.release()
        cv2.destroyAllWindows()
        plt.close("all")

        # Permanently display the final detected frame
        if last_frame is not None:
            clear_output(wait=True)
            plt.figure(figsize=(10, 8))
            plt.imshow(last_frame)
            plt.title("Final Captured Frame (Detections Saved)")
            plt.axis("off")
            plt.show()

            # Optional: save to an image file on disk
            # cv2.imwrite("final_detection.jpg", cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))
            print("Stream ended. Final detection frozen above.")
        else:
            print("No frames were captured.")
```

## Output

<img width="713" height="592" alt="image" src="https://github.com/user-attachments/assets/34e88acc-8a85-4bd6-b283-a0e57b095f06" />


## Result

Thus, the real-time object detection program was successfully executed using YOLOv8m and OpenCV, and the objects from the live webcam feed were accurately detected with bounding boxes and labels.
