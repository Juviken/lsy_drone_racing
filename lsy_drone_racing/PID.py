import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
    
    def reset(self):
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
    
    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output