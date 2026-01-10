import math

class ElectricVehicle:
    def __init__(self):
        # Parameters from Table II of the paper
        self.mass = 2500  # kg (m)
        self.frontal_area = 2.5  # m^2 (Af)
        self.air_density = 1.206  # kg/m^3 (rho)
        self.drag_coeff = 0.28  # (Cd)
        self.rolling_coeff = 0.015  # (f)
        self.wheel_radius = 0.36  # m (R)
        self.battery_capacity = 48000  # Wh (converted 48 kWh to Wh) 
        self.battery_voltage = 360  # V (Voc)
        self.internal_resistance = 0.012  # Ohms (Rb)
        
        # State Variables
        self.velocity = 0.0  # m/s
        self.soc = 0.70  # State of Charge (starts at 70% like in Fig 8d)
        self.position = 0.0 # meters

    def update_physics(self, torque_driver, dt, road_grade=0):
        """
        Calculates new velocity based on Equation (1) from the paper.
        dt: time step in seconds (e.g., 0.1s)
        road_grade: slope in radians (theta)
        """
        # 1. Calculate Forces [cite: 92]
        force_propulsion = torque_driver / self.wheel_radius
        
        force_drag = 0.5 * self.drag_coeff * self.frontal_area * self.air_density * (self.velocity ** 2)
        
        force_gravity = self.mass * 9.81 * math.sin(road_grade)
        
        force_rolling = self.mass * 9.81 * self.rolling_coeff * math.cos(road_grade)
        
        # Net Force
        force_net = force_propulsion - force_drag - force_gravity - force_rolling
        
        # 2. Calculate Acceleration (F = ma -> a = F/m)
        acceleration = force_net / self.mass
        
        # 3. Update Velocity & Position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Prevent negative speed (reversing not supported yet)
        if self.velocity < 0:
            self.velocity = 0

        # 4. Update Battery SOC [cite: 108, 114]
        self._update_energy(torque_driver, self.velocity, dt)

    def _update_energy(self, torque, speed, dt):
        """
        Calculates battery drain based on Eqs (3) and (4).
        """
        # Simple motor efficiency assumption (can be improved with map later)
        motor_efficiency = 0.90 
        
        # Power of motor (P = T * w)
        # w (angular speed) = v / R
        angular_speed = speed / self.wheel_radius
        power_motor = torque * angular_speed
        
        # Battery Power (Pb) - Eq (3)
        if power_motor > 0:
            # Discharging (Driving)
            power_battery = 2 * power_motor / motor_efficiency
        else:
            # Regenerative Braking (Charging)
            power_battery = 2 * power_motor * motor_efficiency

        # Current (Ib) - Derived from P = VI -> I = P/V (Simplified)
        # The paper uses a complex quadratic formula for current (Eq 4), 
        # but for Step 1, P = VI is sufficient to test.
        current = power_battery / self.battery_voltage
        
        # Energy consumed in this step (Wh)
        energy_consumed_wh = (power_battery * dt) / 3600
        
        # Update SOC
        # SOC_new = SOC_old - (Energy_used / Total_Capacity)
        self.soc -= energy_consumed_wh / self.battery_capacity