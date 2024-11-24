import numpy as np
import argparse
import imutils
import time
import cv2
from imutils.video import VideoStream

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("--model", required=True, help="Path to pre-trained Caffe model")
ap.add_argument("--confidence", type=float, default=0.2, help="Minimum confidence threshold")
args = vars(ap.parse_args())

# Initialize class labels and bounding box colors
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start FPS timer
fps_start = time.time()

while True:
    # Grab the frame and resize to a larger window size (600px width)
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Preprocess the frame (blob)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass blob through the network and get predictions
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Display the output frame
    cv2.imshow("Frame", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop FPS timer and print FPS info
fps_end = time.time()
print(f"[INFO] elapsed time: {fps_end - fps_start:.2f}s")

# Cleanup
cv2.destroyAllWindows()
vs.stop()
