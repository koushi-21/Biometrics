import cv2
import numpy as np

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Placeholder for the authenticated user's eye template (grayscale image)
authenticated_eye_template = None
authenticated = False

# Path to save the captured eye image
eye_image_path = r'C:\Users\Admin\Desktop\biopro\pic8.jpg'

# Function to compare eye regions
def compare_eyes(eye_region, template):
    eye_resized = cv2.resize(eye_region, (template.shape[1], template.shape[0]))
    diff = cv2.absdiff(eye_resized, template)
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(diff_thresh)
    total_count = diff_thresh.size
    difference_ratio = non_zero_count / total_count
    return difference_ratio < 0.1

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]

            # If authenticated_eye_template is not set, use the current eye as the template
            if authenticated_eye_template is None:
                authenticated_eye_template = eye_region
                cv2.imwrite(eye_image_path, authenticated_eye_template)  # Save the eye region as an image
                print(f"Captured eye template for authentication and saved as '{eye_image_path}'.")
            else:
                # Compare current eye region with the authenticated template
                if compare_eyes(eye_region, authenticated_eye_template):
                    print("")
                    authenticated = True
                    cv2.putText(frame, "", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Face and Eye Detection', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
