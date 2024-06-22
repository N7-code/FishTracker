
https://github.com/N7-code/FishTracker/assets/173480122/b815ec8b-c927-4dff-8636-e00f42db4996

FishTracker
This project is a video processing library for tracking fish motion in videos. It uses OpenCV for image processing and contour detection to identify and track the position of fish over time. The library performs several steps including resizing the video, applying Gaussian blur, converting to grayscale, and using morphological operations to enhance the fish contours. It also calculates the centroid of the fish and tracks its path, displaying the distance traveled and the speed of the fish in pixels per second.

Features:
Resizes and processes video frames to enhance fish detection.
Applies Gaussian blur and grayscale conversion.
Uses thresholding and masking to isolate fish contours.
Detects fish contours and calculates their centroids.
Tracks the fish's position over time and visualizes its path.
Displays the total distance traveled and speed of the fish.
Dependencies:
OpenCV
NumPy
Usage:
Clone the repository.
Ensure the dependencies are installed.
Run the script with a video file of fish.
