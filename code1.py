import cv2
import numpy as np
import os

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')  # You can use a different mouth Haar cascade

# Path to the saved eye template
eye_template_path = 'pic8.jpg'

# Load the saved template if it exists
if os.path.exists(eye_template_path):
    authenticated_eye_template = cv2.imread(eye_template_path, cv2.IMREAD_GRAYSCALE)
    print("Loaded saved eye template for authentication.")
else:
    authenticated_eye_template = None
    print("No saved template found. Will capture during this session.")

authenticated = False

# Function to compare eye regions
def compare_eyes(eye_region, template):
    # Resize both images to the same size for comparison
    eye_resized = cv2.resize(eye_region, (template.shape[1], template.shape[0]))
    # Calculate the absolute difference between the eye and the template
    diff = cv2.absdiff(eye_resized, template)
    # Threshold to find significant differences
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    # Calculate the percentage of the image that differs
    non_zero_count = np.count_nonzero(diff_thresh)
    total_count = diff_thresh.size
    difference_ratio = non_zero_count / total_count
    # If the difference ratio is small, we can consider it a match
    return difference_ratio < 0.1

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Focus on the face region
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]

            # If no template is saved, capture the current eye as the template
            if authenticated_eye_template is None:
                authenticated_eye_template = eye_region
                # Save the template to a file
                cv2.imwrite(eye_template_path, authenticated_eye_template)
                print(f"Captured and saved eye template for authentication at '{eye_template_path}'")
            else:
                # Compare current eye region with the saved template
                if compare_eyes(eye_region, authenticated_eye_template):
                    print("Authenticated")
                    authenticated = True
                    cv2.putText(frame, "Authenticated", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Detect mouth in the face region
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 20)
        for (mx, my, mw, mh) in mouth:
            if my > h / 2:  # Check if the detected region is in the lower half of the face
                # Draw rectangle around the mouth (lips)
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                cv2.putText(frame, "Lips Detected", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                break  # Consider only the first detected mouth region

    # Display the result
    cv2.imshow('Face, Eye, and Lip Detection', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
