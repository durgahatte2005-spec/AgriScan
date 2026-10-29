# MOCK ROVER DRIVER (For VS Code Testing)
import time

class AgriRover:
    def __init__(self, speed=0.5):
        # We don't initialize real motors here
        self.speed = speed
        print(f"[SIMULATION] 🚜 AgriRover initialized at speed {self.speed}")
        print("[SIMULATION] Ready to receive commands...")

    def forward(self, duration=None):
        print(f"[SIMULATION] ⬆️  Moving FORWARD (Speed: {self.speed})")
        if duration:
            time.sleep(duration)
            self.stop()

    def backward(self, duration=None):
        print(f"[SIMULATION] ⬇️  Moving BACKWARD (Speed: {self.speed})")
        if duration:
            time.sleep(duration)
            self.stop()

    def left(self, duration=None):
        print(f"[SIMULATION] ⬅️  Turning LEFT")
        if duration:
            time.sleep(duration)
            self.stop()

    def right(self, duration=None):
        print(f"[SIMULATION] ➡️  Turning RIGHT")
        if duration:
            time.sleep(duration)
            self.stop()

    def stop(self):
        print("[SIMULATION] 🛑 STOPPING Motors")