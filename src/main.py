import matplotlib.pyplot as plt
from vehicle_dynamics import ElectricVehicle

# 1. Setup Simulation
car = ElectricVehicle()
dt = 0.1  # 100ms time step
simulation_time = 20  # seconds

# Lists to store data for plotting
time_data = []
speed_data = []
soc_data = []

# 2. Run Simulation Loop
for t in range(int(simulation_time / dt)):
    current_time = t * dt
    
    # Simple Driver: Press pedal (200 Nm torque) for 10s, then let go
    if current_time < 10:
        torque_input = 200 # Nm
    else:
        torque_input = 0   # Coasting
        
    # Update Car Physics
    car.update_physics(torque_driver=torque_input, dt=dt)
    
    # Store Data
    time_data.append(current_time)
    speed_data.append(car.velocity)
    soc_data.append(car.soc)

# 3. Plot Results
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Speed (m/s)', color=color)
ax1.plot(time_data, speed_data, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('SOC (State of Charge)', color=color)
ax2.plot(time_data, soc_data, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("EV Dynamics Model Test")
plt.show()