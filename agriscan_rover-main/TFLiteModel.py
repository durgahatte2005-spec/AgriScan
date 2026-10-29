import sys
import types
import time
import cv2
import numpy as np

'''
mobile black : forward
punch : left
palm : right
mobile back : plant
black screen : blocked
'''

# --- PATCH FOR PYTHON 3.13 (Keep this!) ---
if "imp" not in sys.modules:
    sys.modules["imp"] = types.ModuleType("imp")
# ------------------------------------------

# Import TensorFlow (or TFLite Runtime)
try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# Import your custom Driver Class
# NOTE: Ensure the file is named 'Rover_Movement.py' and class is 'AgriRover'
from Rover_Movement import AgriRover

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.7  # Only move if AI is 70% sure

# Initialize the Rover Body
rover = AgriRover(speed=0.5)

# Initialize the Camera
cap = cv2.VideoCapture(0)
cap.set(3, 320) # Width
cap.set(4, 240) # Height

# ==========================================
# AI SETUP
# ==========================================
print(f"[AI] Loading Model: {MODEL_PATH}...")

# Load Labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load Model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']

print("[AI] System Online. Press Ctrl+C to Stop.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_prediction(frame):
    """
    Takes a camera frame, processes it, and returns the
    Label and Confidence score from the TFLite model.
    """
    # 1. Convert BGR (OpenCV) to RGB (AI expects this)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Resize to 224x224 (Standard Teachable Machine size)
    img_resized = cv2.resize(img_rgb, (width, height))

    # 3. Add batch dimension (1, 224, 224, 3)
    input_data = np.expand_dims(img_resized, axis=0)

    # 4. Normalize (Teachable Machine standard: -1 to 1 or 0 to 1)
    # Most float models use this normalization: (pixel - 127.5) / 127.5
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # 5. Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 6. Get Result
    max_index = np.argmax(output_data[0])
    prediction_label = labels[max_index]
    confidence = output_data[0][max_index]

    return prediction_label, confidence

# ==========================================
# MAIN LOOP
# ==========================================
try:
    while True:
        # 1. Capture Image
        ret, frame = cap.read()
        if not ret:
            print("[Error] Camera failed.")
            break

        # 2. Get AI Decision
        label, conf = get_prediction(frame)
        
        # Display output on screen (optional, good for debugging)
        debug_text = f"{label} ({conf*100:.1f}%)"
        cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AgriScan AI", frame)
        cv2.waitKey(1)

        # 3. EXECUTE ROVER MOVEMENT
        # Adjust these strings to match EXACTLY what you typed in Teachable Machine
        
        if conf < CONFIDENCE_THRESHOLD:
            print(f"Unsure ({conf:.2f}). Stopping.")
            rover.stop()
        
        elif "Path" in label or "Forward" in label:
            print(f"Action: FORWARD | {debug_text}")
            rover.forward()
            
        elif "Left" in label:
            print(f"Action: LEFT | {debug_text}")
            rover.left()
            
        elif "Right" in label:
            print(f"Action: RIGHT | {debug_text}")
            rover.right()
            
        elif "Plant" in label or "Stop" in label:
            print(f"Action: TARGET REACHED | {debug_text}")
            rover.stop()
            # Here you would call your Gemini Disease Detection function
            time.sleep(2) # Wait for "scanning"
            
        elif "Blocked" in label:
            print(f"Action: OBSTACLE | {debug_text}")
            rover.stop()
            time.sleep(0.5)
            rover.backward(1.0) # Back up for 1 second
            rover.right(1.0)    # Turn around
            
        else:
            rover.stop()

        # Small delay to prevent CPU overheating
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[AI] Shutting Down.")
    rover.stop()
    cap.release()
    cv2.destroyAllWindows()