import numpy as np

class PIDController:
    #Basic PID controller with the ability to reset the integral and compute the output
    def __init__(self, kp, ki, kd, dt): #Takes the proportional, integral, derivative gains and the time step
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