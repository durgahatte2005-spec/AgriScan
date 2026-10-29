import cv2
import numpy as np
import time
import sys
import types

# --- MOCK DRIVER IMPORT ---
# This looks for Rover_Movement.py in the same folder
from demo import AgriRover

# --- TENSORFLOW SETUP ---
# On PC, we usually use full 'tensorflow'. On Pi, we use 'tflite_runtime'.
# This block handles both automatically.
try:
    import tensorflow.lite as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        print("[ERROR] TensorFlow not found. Please run: pip install tensorflow")
        sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.7

# Initialize the (Fake) Rover
rover = AgriRover(speed=0.5)

# Initialize Webcam
print("[SYSTEM] Opening Webcam...")
cap = cv2.VideoCapture(0) # 0 is usually the default laptop webcam

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    sys.exit(1)

# ==========================================
# AI SETUP
# ==========================================
print(f"[AI] Loading Model: {MODEL_PATH}...")

try:
    # Load Labels
    with open(LABELS_PATH, "r") as f:
        # Strip newline characters
        labels = [line.strip() for line in f.readlines()]

    # Load Model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, height, width, _ = input_details[0]['shape'] # Usually 224, 224

    print("[AI] System Online. Press 'q' to Quit.")

except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("Check if 'model.tflite' and 'labels.txt' are in this folder!")
    sys.exit(1)

# ==========================================
# HELPER FUNCTION
# ==========================================
def get_prediction(frame):
    # 1. Convert BGR (OpenCV) to RGB (AI expects this)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Resize to 224x224
    img_resized = cv2.resize(img_rgb, (width, height))

    # 3. Add batch dimension (1, 224, 224, 3)
    input_data = np.expand_dims(img_resized, axis=0)

    # 4. Normalize (0 to 1 or -1 to 1)
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
            print("[Error] Frame capture failed.")
            break

        # 2. Get AI Decision
        label, conf = get_prediction(frame)

        # 3. Display Debug Info on Screen
        # Clean up label (remove numbers if Teachable Machine added them, e.g. "0 Path")
        clean_label = label.split(" ", 1)[-1] if " " in label else label
        
        text = f"{clean_label}: {conf*100:.1f}%"
        color = (0, 255, 0) if conf > CONFIDENCE_THRESHOLD else (0, 0, 255)
        
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("AgriScan PC Simulation", frame)

        # 4. DECISION LOGIC (The Brain)
        if conf < CONFIDENCE_THRESHOLD:
            print(f"🤔 Unsure ({conf:.2f})... Stopping.")
            rover.stop()

        elif "path" in label:
            print(f"✅ Action: FORWARD ({clean_label})")
            rover.forward()

        elif "left" in label:
            print(f"⬅️ Action: LEFT ({clean_label})")
            rover.left()

        elif "right" in label:
            print(f"➡️ Action: RIGHT ({clean_label})")
            rover.right()

        elif "plant" in label:
            print(f"🌿 Action: PLANT DETECTED ({clean_label})")
            rover.stop()
            # On PC, we just print this. On Pi, this calls Gemini.
            print("   [Mock] Calling Gemini API for disease analysis...")
            time.sleep(1) # Simulate delay

        elif "block" in label:
            print(f"🧱 Action: BLOCKED ({clean_label})")
            rover.stop()
            rover.backward(0.5)

        else:
            rover.stop()

        # 5. Quit Logic
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Slow down simulation slightly so logs are readable
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[System] Simulation Stopped.")

cap.release()
cv2.destroyAllWindows()