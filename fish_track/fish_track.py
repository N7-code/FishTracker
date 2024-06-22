import cv2
import numpy as np
import time
import math
from findPoints import *

# Initial time
initial_time = round(time.time(), 2)

# Import video
video = cv2.VideoCapture('fish.mp4')

# Set recording video parameters
fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.6)
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.6)
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Store fish positions
pts = []

while video.isOpened():
    # Current time
    current_time = round(time.time(), 2)
    ret, frame = video.read()
    
    if ret:
        # Resize
        frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
        original = frame.copy()
       
        # Gaussian Blur
        img_Gaussblurred = cv2.GaussianBlur(frame, (3, 3), 0)
        #draw('guess_img', img_blurred)

        # Grayscale and Binarization
        img_gray = cv2.cvtColor(img_Gaussblurred, cv2.COLOR_BGR2GRAY)
        #draw('gray_img', img_gray)
        _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)
        _, img_eye = cv2.threshold(img_gray, 65, 255, cv2.THRESH_BINARY_INV)
        #draw('img_binary', img_binary)
        #draw('img_eye', img_eye)
        
        # Create mask
        mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        center_x = mask.shape[1] // 2
        center_y = mask.shape[0] // 2
        radius = 175
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        #draw('mask', mask)
        img_and = cv2.bitwise_and(img_binary, mask)
        img_eye_and = cv2.bitwise_and(img_eye, mask)
        #draw('img_and', img_and)
        #draw('img_eye_and', img_eye_and)

        # Morphological processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_dilate = cv2.dilate(img_and, kernel, iterations=2) # Dilation
        #draw('img_dilate', img_dilate)

        # Denoising the eyes
        img_eye_dilate = cv2.dilate(img_eye_and, kernel, iterations=2) # Dilation
        #draw(' img_eye_dilate',  img_eye_dilate)

        # Find contours
        cnts, _ = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #eye_cnts, _ = cv2.findContours(img_eye_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            # Calculate the centroid of the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                # Calculate the centroid coordinates of the contour
                fish_cx = int(M["m10"] / M["m00"])
                fish_cy = int(M["m01"] / M["m00"])
                
                # Find the points of the contour
                contour_points = c.squeeze()
                
                # Find the two points that are the farthest apart
                max_distance = 0
                head_index = tail_index = (0, 0)
                for i in range(len(contour_points)):
                    for j in range(i + 1, len(contour_points)):
                        distance = np.linalg.norm(contour_points[i] - contour_points[j])
                        if distance > max_distance:
                            max_distance = distance
                            head_index = i
                            tail_index = j

                # Calculate the center point of the ellipse
                center = ((contour_points[head_index][0] + contour_points[tail_index][0]) // 2, (contour_points[head_index][1] + contour_points[tail_index][1]) // 2)

                # Calculate the angle of the line segment between the two farthest points
                angle = np.arctan2(contour_points[tail_index][1] - contour_points[head_index][1],
                                contour_points[tail_index][0] - contour_points[head_index][0]) * 180 / np.pi

                # Draw the ellipse with the specified rotation angle
                cv2.ellipse(frame, center, (int(max_distance/2), int(max_distance/4)), angle, 0, 360, (0, 255, 0), 2)

                
        # Find contours of the eye area
        eye_cnts, _ = cv2.findContours(img_eye_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize head positions
        head_positions = []

        for c in eye_cnts:
            # Calculate the centroid of the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                # Calculate the centroid coordinates of the contour
                eye_cx = int(M["m10"] / M["m00"])
                eye_cy = int(M["m01"] / M["m00"])
                head_positions.append((eye_cx, eye_cy))

        # Find the center position of all centroid points
        if head_positions: 
            center_x = sum([pos[0] for pos in head_positions]) // len(head_positions)
            center_y = sum([pos[1] for pos in head_positions]) // len(head_positions)

            # Draw a blue circle to mark this center position
            cv2.circle(frame, (center_x, center_y), 10, (255, 0, 0), 2)

        distance = []
        red = (255, 0, 255)
        # Draw the fish trajectory
        pts.append((fish_cx, fish_cy))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], red, 2)
            point1 = pts[i]
            point2 = pts[i + 1]
            distance.append(np.linalg.norm(np.array(point2) - np.array(point1)))
        
        all_distance = round(sum(distance))
        all_time = round(current_time - initial_time, 2)
        speed = round(all_distance / all_time, 2)
        cv2.putText(frame, f'Time: ' + str(all_time), (248, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 2)
        cv2.putText(frame, f'(S)', (355, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 2)
        cv2.putText(frame, f'Distance: ' + str(all_distance), (218, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 2)
        cv2.putText(frame, f'(pixels)', (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 2)
        cv2.putText(frame, f'Speed: ' + str(speed), (238, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 2)
        cv2.putText(frame, f'(pixels/S)', (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 2)
        cv2.putText(frame, f'Centroid: ({fish_cx}, {fish_cy})', (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 2)
        cv2.putText(frame, f'(pixels)', (355, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 0, 0), 2)

        output.write(frame)
        cv2.imshow('original', original)
        cv2.moveWindow('original', 0, 10)
        cv2.imshow('img_and', img_and)
        cv2.moveWindow('img_and', 500, 10)
        cv2.imshow('fish', frame)
        cv2.moveWindow('fish', 1000, 10)
        
    if cv2.waitKey(10) == 27:
        break

video.release()
output.release()
cv2.destroyAllWindows()
