import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from vehicle_dynamics import ElectricVehicle

class MPCController:
    def __init__(self, model_path='models/traffic_lstm.h5'):
        # Load the AI we trained in Phase 2
        try:
            self.ai_model = tf.keras.models.load_model(model_path)
        except:
            print("❌ Error: Could not load model. Using dummy prediction.")
            self.ai_model = None
        
        # MPC Parameters
        self.horizon = 20
        self.dt = 0.5
        
        # --- TUNED WEIGHTS ---
        self.w_safety = 1000.0   # CRITICAL: Do not crash
        self.w_tracking = 20.0   # NEW: Match leader's speed
        self.w_energy = 0.01     # Save energy (lowered to prioritize moving)
        self.w_comfort = 10.0    # Smooth driving
        
        self.safe_distance = 15.0 # meters

    def predict_leader_trajectory(self, recent_speed_history):
        if self.ai_model is None:
            return np.array([recent_speed_history[-1]] * self.horizon)
        input_data = np.array(recent_speed_history).reshape(1, 100, 1)
        predicted_speeds = self.ai_model.predict(input_data, verbose=0)
        return predicted_speeds[0]

    def optimize_control(self, ego_vehicle, leader_future_speeds, current_gap):
        # Initial guess: Try to match the leader's current speed (converted to torque approx)
        # Force = mass * accel + drag... approximate steady state torque ~ 200
        initial_guess = np.full(self.horizon, 200.0)
        
        bounds = [(-3000, 3000) for _ in range(self.horizon)] # Increased torque limits

        def cost_function(torque_plan):
            cost = 0
            v_ego = ego_vehicle.velocity
            p_ego = 0
            p_leader = current_gap
            
            for i in range(self.horizon):
                # 1. Physics Update
                accel = torque_plan[i] / (ego_vehicle.mass * ego_vehicle.wheel_radius)
                v_ego += accel * self.dt
                p_ego += v_ego * self.dt
                
                v_leader = leader_future_speeds[i]
                p_leader += v_leader * self.dt
                
                gap = p_leader - p_ego
                
                # --- COST TERMS ---
                
                # A. Tracking (NEW): Match Leader Speed
                # If we don't have this, the car stops to save energy!
                cost += self.w_tracking * (v_ego - v_leader)**2

                # B. Safety: Massive penalty if too close
                if gap < self.safe_distance:
                    cost += self.w_safety * (self.safe_distance - gap)**2
                
                # C. Energy: Minimize torque usage
                cost += self.w_energy * abs(torque_plan[i])
                
                # D. Comfort: Minimize Jerk
                if i > 0:
                    jerk = abs(torque_plan[i] - torque_plan[i-1])
                    cost += self.w_comfort * jerk

            return cost

        result = minimize(cost_function, initial_guess, bounds=bounds, method='SLSQP')
        return result.x[0]