# Real-Time-Object-Detection-With-OpenCV

This project aims to do real-time object detection through a laptop camera or webcam using OpenCV and MobileNetSSD. The idea is to loop over each frame of the video stream, detect objects bound each detection in a box and also to give the real time audio output of the detected object.

### Follow the steps below to run the code

**Clone the Repository:**
```
git clone https://github.com/agnur-bharath/Real-Time-Object-Detection-With-OpenCV.git
cd Real-Time-Object-Detection-With-OpenCV
```
**Install the Required Packages:**

```
pip install opencv-python
pip install opencv-contrib-python
pip install opencv-python-headless
pip install opencv-contrib-python-headless
pip install matplotlib
pip install imutils
```

Make sure to download and install opencv and and opencv-contrib releases for OpenCV 3.3. This ensures that the deep neural network (dnn) module is installed. You must have OpenCV 3.3 (or newer) to run this code.

**Run the following command to execute the code:**

To run the code for object detection without Audio output use the below command in your terminal
```
python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
```

To run the code for object detection and audio output use the below command in your terminal
```
python real_time_object_detection_with_voice.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel
```