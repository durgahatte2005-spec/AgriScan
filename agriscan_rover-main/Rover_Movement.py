from gpiozero import Robot
from time import sleep

# --- CONFIGURATION (EDIT PINS HERE) ---
LEFT_MOTOR_PIN_1 = 23
LEFT_MOTOR_PIN_2 = 24
RIGHT_MOTOR_PIN_1 = 27
RIGHT_MOTOR_PIN_2 = 22
# --------------------------------------

class AgriRover:
    def __init__(self, speed=0.5):
        """
        Initialize the Rover Motors using the pins defined above.
        """
        try:
            # We tell the Robot object specifically which pins to use here
            self.robot = Robot(
                left=(LEFT_MOTOR_PIN_1, LEFT_MOTOR_PIN_2), 
                right=(RIGHT_MOTOR_PIN_1, RIGHT_MOTOR_PIN_2)
            )
            self.speed = speed
            print(f"[AgriRover] Initialized on Pins Left:{LEFT_MOTOR_PIN_1}/{LEFT_MOTOR_PIN_2} Right:{RIGHT_MOTOR_PIN_1}/{RIGHT_MOTOR_PIN_2}")
        except Exception as e:
            print(f"[AgriRover] Motor Init Failed: {e}")
            self.robot = None

    def forward(self, duration=None):
        """Move Forward."""
        if self.robot:
            self.robot.forward(self.speed)
            if duration:
                sleep(duration)
                self.stop()

    def backward(self, duration=None):
        """Move Backward."""
        if self.robot:
            self.robot.backward(self.speed)
            if duration:
                sleep(duration)
                self.stop()

    def left(self, duration=None):
        """Turn Left."""
        if self.robot:
            self.robot.left(self.speed)
            if duration:
                sleep(duration)
                self.stop()

    def right(self, duration=None):
        """Turn Right."""
        if self.robot:
            self.robot.right(self.speed)
            if duration:
                sleep(duration)
                self.stop()

    def stop(self):
        """Stop all motors."""
        if self.robot:
            self.robot.stop()