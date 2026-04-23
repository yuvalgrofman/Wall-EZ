import time
import socket
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

# Initialize the pigpio factory for precise hardware PWM
factory = PiGPIOFactory()

# --- Wheel Servos (Continuous) ---
PIN_LF = 17  # Physical Pin 11
PIN_LB = 4   # Physical Pin 7
PIN_RB = 26  # Physical Pin 37
PIN_RF = 27  # Physical Pin 13

# --- Arm / Placement Servos ---
PIN_ARM1 = 21 # Physical Pin 40 - A
PIN_ARM2 = 20 # Physical Pin 38 - A
PIN_ARM3 = 23 # Physical Pin 36 - B

# --- ESC ---
PIN_ESC = 18  # Physical Pin 12

# Initialize all servos
servo_lf = Servo(PIN_LF, pin_factory=factory)
servo_lb = Servo(PIN_LB, pin_factory=factory)
servo_rb = Servo(PIN_RB, pin_factory=factory)
servo_rf = Servo(PIN_RF, pin_factory=factory)

servo_arm1 = Servo(PIN_ARM1, pin_factory=factory)
servo_arm2 = Servo(PIN_ARM2, pin_factory=factory)
servo_arm3 = Servo(PIN_ARM3, pin_factory=factory)

# Group them for easier iteration
wheels = [servo_lf, servo_lb, servo_rb, servo_rf]
arms = [servo_arm1, servo_arm2, servo_arm3]

esc = Servo(PIN_ESC, pin_factory=factory)

# --- Network Configuration ---
UDP_IP = "0.0.0.0"
UDP_PORT = 4210

def apply_esc_microsec(val):
    if val < 1000:
        val = 1000
    elif val > 2000:
        val = 2000
        
    mapped_value = (val - 1500) / 500.0
    esc.value = mapped_value
    print(f"ESC set to {val}us -> Value: {mapped_value:.2f}")


###################################################
# ############# MAKE SURE ENGINE IS OFF ###########
###################################################
apply_esc_microsec(1000)
###################################################
# ############# MAKE SURE ENGINE IS OFF ###########
###################################################

def map_degree_to_value(degree):
    """Maps a 0-180 degree value to the -1.0 to 1.0 range."""
    return (degree - 90) / 90.0

def process_command(cmd):
    cmd = cmd.strip()
    print(f"Processing command: {cmd}")
    
    if cmd == "RIGHT":
        # # right wheels backward, left wheels forward
        # for wheel in wheels:
        #     wheel.value = map_degree_to_value(180)

        # right wheels stop, left wheels forward
        servo_lf.value = map_degree_to_value(180)
        servo_lb.value = map_degree_to_value(180)
        servo_rf.value = map_degree_to_value(180)
        servo_rb.value = map_degree_to_value(180)

    elif cmd == "LEFT":
        for wheel in wheels:
            wheel.value = map_degree_to_value(0)
            
    elif cmd == "BWD":
        # Left wheels backward, Right wheels forward
        servo_lf.value = map_degree_to_value(0)
        servo_lb.value = map_degree_to_value(0)
        servo_rf.value = map_degree_to_value(180)
        servo_rb.value = map_degree_to_value(180)
        
    elif cmd == "FWD":
        # Left wheels forward, Right wheels backward
        servo_lf.value = map_degree_to_value(180)
        servo_lb.value = map_degree_to_value(180)
        servo_rf.value = map_degree_to_value(40)
        servo_rb.value = map_degree_to_value(40)
            
    elif cmd == "STOP":
        for wheel in wheels:
            wheel.value = map_degree_to_value(90)
            
    elif cmd == "ARM_UP":
        servo_arm1.value = map_degree_to_value(120)
    
    elif cmd == "ARM_DOWN":
        servo_arm1.value = map_degree_to_value(180)
        servo_arm2.value = map_degree_to_value(180)
        servo_arm3.value = map_degree_to_value(0)

    elif cmd == "ARM_STOP":
        servo_arm1.value = map_degree_to_value(90)
        servo_arm2.value = map_degree_to_value(90)
        servo_arm3.value = map_degree_to_value(90)
        
    elif cmd.startswith("ESC:"):
        try:
            val = int(cmd.split(":")[1])
            apply_esc_microsec(val)
        except ValueError:
            print("Invalid ESC command format")

if __name__ == "__main__":

    # before execution run these commands in the terminal
    # source my_env/bin/activate
    # sudo pigpiod

    try:
        # MUST!!!  apply_esc_microsec(1000)  # Neutral for ESC time.sleep(1)  # Allow ESC to initialize
        apply_esc_microsec(1000)  # Neutral for ESC

        apply_esc_microsec(1500)  # Start at neutral
        time.sleep(1)
        apply_esc_microsec(1950)  # Full forward
        time.sleep(3)

        # process_command("ARM_DOWN")  
        # time.sleep(0.5)
        # process_command("ARM_STOP")


        process_command("FWD")
        time.sleep(1)
        process_command("RIGHT")
        time.sleep(0.07 * 5)
        process_command("STOP")
        time.sleep(0.3)
        process_command("FWD")
        time.sleep(1)
        process_command("STOP")
        time.sleep(2)

        apply_esc_microsec(1000)  # Neutral for ESC

    finally:
        process_command("STOP")
        apply_esc_microsec(1000)  # Neutral for ESC