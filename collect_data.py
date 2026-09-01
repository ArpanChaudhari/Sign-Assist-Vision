import cv2
import mediapipe as mp
import csv
import os

# 1. Set up MediaPipe hand detection model

# Get required MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Set options for hand detection
options = HandLandmarkerOptions(
    # Path of the MediaPipe hand detection model
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    # We are detecting hands from individual images
    running_mode=RunningMode.IMAGE,
    # Detect only one hand
    num_hands=1,
)


# 2. Create a CSV file to store hand landmarks and gesture labels

csv_file_name = "hand_data.csv"


# Create the CSV file only if it does not already exist
if not os.path.exists(csv_file_name):

    # Open the file in write mode
    with open(csv_file_name, mode="w", newline="") as file:

        writer = csv.writer(file)

        # First column stores the gesture label
        headers = ["label"]

        # Each hand has 21 landmarks. Every landmark has x, y, and z coordinates
        for landmark_number in range(21):

            headers.append(f"x{landmark_number}")
            headers.append(f"y{landmark_number}")
            headers.append(f"z{landmark_number}")

        # Write column names to the CSV file
        writer.writerow(headers)


# 3. Start capturing video from the webcam

camera = cv2.VideoCapture(0)

print("Press A-Z or 0-9 to save a gesture.")
print("Press '-' to save DEL.")
print("Press '=' to save SPACE.")
print("Press 'Q' to quit.")


# 4. Detect hand landmarks in a loop and save them to the CSV file

with HandLandmarker.create_from_options(options) as hand_detector:

    while True:

        # Read one frame from the webcam
        success, frame = camera.read()

        # Stop if the camera cannot read a frame
        if not success:
            break

        # Flip the camera image horizontally. This makes the camera behave like a mirror
        frame = cv2.flip(frame, 1)

        # Convert OpenCV image from BGR to RGB. MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the NumPy image into a MediaPipe Image
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        detection_result = hand_detector.detect(mediapipe_image)

        # Check if any hands were detected
        if detection_result.hand_landmarks:

            # Loop through detected hands
            for hand in detection_result.hand_landmarks:

                # 5. Draw landmarks on the camera image

                for landmark in hand:

                    # Convert normalized coordinates (0 to 1)
                    # into actual pixel coordinates
                    x_position = int(landmark.x * frame.shape[1])

                    y_position = int(landmark.y * frame.shape[0])

                    # Draw a green circle on every landmark
                    cv2.circle(frame, (x_position, y_position), 5, (0, 255, 0), -1)

                # 6. Make landmark coordinates relative to the wrist and save them to the CSV file

                # Landmark 0 is the wrist
                wrist_x = hand[0].x
                wrist_y = hand[0].y
                wrist_z = hand[0].z

                # This list will store all hand coordinates
                gesture_data = []

                # Calculate coordinates relative to the wrist
                for landmark in hand:

                    relative_x = landmark.x - wrist_x
                    relative_y = landmark.y - wrist_y
                    relative_z = landmark.z - wrist_z

                    # Add x, y, and z values to the list
                    gesture_data.extend([relative_x, relative_y, relative_z])

                    # (This goes right after Step 6 where you finish calculating relative_z)

        # 7. Display Camera Image with Landmarks
        cv2.imshow("Data Collector", frame)

        # 8. Listen for a single keyboard press (Pauses for 1 millisecond)
        key = cv2.waitKey(1) & 0xFF

        # 9. Check if we need to save data (Only works if a hand was detected)
        if detection_result.hand_landmarks:

            # A-Z or 0-9
            if ord("a") <= key <= ord("z") or ord("0") <= key <= ord("9"):
                gesture_label = chr(key).upper()
                with open(csv_file_name, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([gesture_label] + gesture_data)
                print(f"Saved gesture: {gesture_label}")

            # "-" KEY = DEL
            elif key == ord("-"):
                with open(csv_file_name, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["DEL"] + gesture_data)
                print("Saved gesture: DEL")

            # "=" KEY = SPACE
            elif key == ord("="):
                with open(csv_file_name, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["SPACE"] + gesture_data)
                print("Saved gesture: SPACE")

        # 10. Press "Q" to quit the program (Works even if no hand is detected)
        if key == ord("q"):
            break


# 11. Clean up and release resources
camera.release()
cv2.destroyAllWindows()
