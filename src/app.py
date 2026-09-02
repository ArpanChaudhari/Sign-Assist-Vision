import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

# 1. Rebuild the model structure
class SignModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 2. Load classes and trained model
classes = np.load('models/classes.npy', allow_pickle=True)
model = SignModel(len(classes))
model.load_state_dict(torch.load('models/model.pth'))
model.eval()

# 3. Setup the New MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

# --- NEW KEYBOARD VARIABLES ---
sentence = ""           # The full typed sentence
last_pred = ""          # What the AI currently sees
frames_held = 0         # How long you've held the sign
REQUIRED_FRAMES = 40    # How many frames to hold before it "types" (adjust to type faster/slower)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for lm in hand_landmarks:
                    x_px = int(lm.x * img.shape[1])
                    y_px = int(lm.y * img.shape[0])
                    cv2.circle(img, (x_px, y_px), 5, (0, 255, 0), -1)
                    
                base_x, base_y, base_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
                
                row = []
                for lm in hand_landmarks:
                    row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                
                tensor_input = torch.FloatTensor([row])
                with torch.no_grad():
                    output = model(tensor_input)
                    prediction_index = torch.argmax(output, dim=1).item()
                    predicted_label = classes[prediction_index]
                    
                    # --- SENTENCE BUILDER LOGIC ---
                    if predicted_label == last_pred:
                        frames_held += 1
                    else:
                        frames_held = 0
                        last_pred = predicted_label
                    
                    # If held long enough, TYPE IT!
                    if frames_held == REQUIRED_FRAMES:
                        if predicted_label == "SPACE":
                            sentence += " "
                        elif predicted_label == "DEL":
                            sentence = sentence[:-1] # Remove last character
                        else:
                            sentence += predicted_label
                            
                        # Add a cooldown so it doesn't instantly type 'AAAAA'
                        # (To type double letters like 'LL', just drop your hand for a second)
                        frames_held = -10
                    
                    # --- VISUALS ---
                    # 1. Print the live prediction
                    cv2.putText(img, f'Seeing: {predicted_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # 2. Draw the typing progress bar
                    if frames_held > 0:
                        progress_width = int((frames_held / REQUIRED_FRAMES) * 200)
                        cv2.rectangle(img, (10, 60), (10 + progress_width, 75), (0, 255, 0), cv2.FILLED)
                        cv2.rectangle(img, (10, 60), (210, 75), (255, 255, 255), 2) # border
        
        else:
            # If no hand is on screen, reset the progress bar
            frames_held = 0
            last_pred = ""

        # --- DRAW THE SENTENCE BOARD ---
        # Draw a black box at the bottom of the screen
        cv2.rectangle(img, (0, img.shape[0] - 60), (img.shape[1], img.shape[0]), (0, 0, 0), cv2.FILLED)
        # Write the typed sentence
        cv2.putText(img, sentence, (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow("Sign Language Translator", img)
        
        # Press 'C' to clear the sentence, 'Q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            sentence = ""
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()