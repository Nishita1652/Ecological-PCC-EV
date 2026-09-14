import numpy as np
import matplotlib.pyplot as plt
from vehicle_dynamics import ElectricVehicle
from mpc_controller import MPCController

class SimpleDriver:
    def compute_torque(self, ego_speed, leader_speed, gap, dt):
        safe_following_distance = max(5.0, ego_speed * 2.0)
        
        if gap < safe_following_distance:
            return -2500 
        elif gap > safe_following_distance + 10:
            return 1500
        else:
            speed_diff = leader_speed - ego_speed
            return speed_diff * 500

def get_tough_traffic_profile(duration=60, dt=0.5):
    steps = int(duration / dt)
    t = np.linspace(0, duration, steps)
    v_leader = 15 + 10 * np.sin(t / 10) 
    
    for i, time in enumerate(t):
        if 25 < time < 35:
            v_leader[i] -= 10
            
    v_leader += np.random.normal(0, 0.5, steps)
    v_leader = np.clip(v_leader, 0, 35)
    return t, v_leader

def run_race():
    dt = 0.5
    duration = 60
    t_data, leader_speed_profile = get_tough_traffic_profile(duration, dt)
    steps = len(t_data)

    eco_car = ElectricVehicle()
    mpc_controller = MPCController()
    
    dumb_car = ElectricVehicle()
    simple_driver = SimpleDriver()

    hist_eco_speed, hist_dumb_speed = [], []
    hist_eco_soc, hist_dumb_soc = [], []
    hist_eco_gap, hist_dumb_gap = [], []
    leader_actual_speeds = []

    ai_speed_buffer = [20.0] * 100
    current_gap_eco = 30.0
    current_gap_dumb = 30.0

    for k in range(steps - 20):
        leader_v = leader_speed_profile[k]
        ai_speed_buffer.pop(0)
        ai_speed_buffer.append(leader_v)
        
        pred_traj = mpc_controller.predict_leader_trajectory(ai_speed_buffer)
        torque_eco = mpc_controller.optimize_control(eco_car, pred_traj, current_gap_eco)
        eco_car.update_physics(torque_eco, dt)
        current_gap_eco += (leader_v - eco_car.velocity) * dt
        
        torque_dumb = simple_driver.compute_torque(dumb_car.velocity, leader_v, current_gap_dumb, dt)
        dumb_car.update_physics(torque_dumb, dt)
        current_gap_dumb += (leader_v - dumb_car.velocity) * dt

        leader_actual_speeds.append(leader_v)
        hist_eco_speed.append(eco_car.velocity)
        hist_dumb_speed.append(dumb_car.velocity)
        hist_eco_soc.append(eco_car.soc)
        hist_dumb_soc.append(dumb_car.soc)
        hist_eco_gap.append(current_gap_eco)
        hist_dumb_gap.append(current_gap_dumb)

    energy_eco_wh = (0.70 - eco_car.soc) * eco_car.battery_capacity
    energy_dumb_wh = (0.70 - dumb_car.soc) * dumb_car.battery_capacity
    savings_percent = ((energy_dumb_wh - energy_eco_wh) / energy_dumb_wh) * 100

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0,0].plot(leader_actual_speeds, 'k--', label='Leader (Traffic)', alpha=0.7)
    axs[0,0].plot(hist_dumb_speed, 'r', label='Dumb Driver', alpha=0.6)
    axs[0,0].plot(hist_eco_speed, 'g', label='Eco-MPC (Yours)', linewidth=2)
    axs[0,0].set_title("Velocity Profile")
    axs[0,0].legend()
    
    axs[0,1].plot(hist_dumb_soc, 'r', label='Dumb SOC')
    axs[0,1].plot(hist_eco_soc, 'g', label='Eco SOC')
    axs[0,1].set_title(f"Savings: {savings_percent:.1f}%")
    axs[0,1].legend()
    
    axs[1,0].plot(hist_dumb_gap, 'r', label='Dumb Gap')
    axs[1,0].plot(hist_eco_gap, 'g', label='Eco Gap')
    axs[1,0].axhline(15, color='k', linestyle=':', label='Min Safe Dist')
    axs[1,0].set_title("Safety Gap")
    axs[1,0].legend()
    
    axs[1,1].bar(['Dumb Driver', 'Eco-MPC'], [energy_dumb_wh, energy_eco_wh], color=['red', 'green'])
    axs[1,1].set_title("Total Energy Consumed (Wh)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_race()