import math

class ElectricVehicle:
    def __init__(self):
        # Parameters from Table II of the paper
        self.mass = 2500  # kg
        self.frontal_area = 2.5  # m^2
        self.air_density = 1.206  # kg/m^3
        self.drag_coeff = 0.28  
        self.rolling_coeff = 0.015 
        self.wheel_radius = 0.36  # m
        self.battery_capacity = 48000  # Wh
        self.battery_voltage = 360  # V
        self.internal_resistance = 0.012  # Ohms
        
        # State Variables
        self.velocity = 0.0  # m/s
        self.soc = 0.70  # Start at 70%
        self.position = 0.0 # meters
        
        # --- NEW: Initialize the counters ---
        self.total_energy_consumed = 0.0 # Wh
        self.total_distance = 0.0 # m

    def update_physics(self, torque_driver, dt, road_grade=0):
        # 1. Calculate Forces
        force_propulsion = torque_driver / self.wheel_radius
        force_drag = 0.5 * self.drag_coeff * self.frontal_area * self.air_density * (self.velocity ** 2)
        force_gravity = self.mass * 9.81 * math.sin(road_grade)
        force_rolling = self.mass * 9.81 * self.rolling_coeff * math.cos(road_grade)
        
        # Net Force
        force_net = force_propulsion - force_drag - force_gravity - force_rolling
        
        # 2. Update Motion
        acceleration = force_net / self.mass
        self.velocity += acceleration * dt
        
        # Prevent reversing (Simple model)
        if self.velocity < 0:
            self.velocity = 0
            
        self.position += self.velocity * dt
        self.total_distance += self.velocity * dt

        # 3. Update Energy
        self._update_energy(torque_driver, self.velocity, dt)

    def _update_energy(self, torque, speed, dt):
        """
        Calculates battery drain AND Regeneration.
        """
        motor_efficiency = 0.92
        regen_efficiency = 0.75
        
        # Power = Torque * Angular Speed
        angular_speed = speed / self.wheel_radius
        mechanical_power = torque * angular_speed
        
        if mechanical_power > 0:
            # DRIVING (Discharging)
            electrical_power = mechanical_power / motor_efficiency
        else:
            # BRAKING (Charging)
            electrical_power = mechanical_power * regen_efficiency
            
        # Update SOC
        energy_wh = (electrical_power * dt) / 3600
        self.soc -= energy_wh / self.battery_capacity
        
        # Track total stats (only count consumption, not regen, for the 'consumed' metric)
        if energy_wh > 0:
            self.total_energy_consumed += energy_wh