# Ecological Predictive Cruise Control (Eco-PCC)

This repository implements a predictive, energy-optimizing cruise control system for Electric Vehicles (EVs). By pairing a Deep Learning traffic forecasting model with a multi-objective Model Predictive Controller (MPC), the system dynamically calculates torque trajectories to minimize battery drain while strictly maintaining safety boundaries.

Developed as a high-fidelity proof-of-concept for automotive R&D, this project demonstrates how shifting from reactive following to predictive pacing can effectively harvest kinetic energy and reduce aerodynamic drag without compromising passenger comfort.

**Core Architecture**
* **Forecasting Engine (Keras 3):** A stacked Long Short-Term Memory (LSTM) network processes a 100-step historical speed buffer to predict the leader vehicle's trajectory across a 20-step future horizon. 
* **Model Predictive Control (SciPy):** An SLSQP optimizer evaluates the predicted horizon against a multi-objective cost function, actively balancing speed tracking, safety gap maintenance (15m–35m), jerk minimization, and energy conservation.
* **Asymmetric EV Physics:** The vehicle dynamics engine calculates realistic longitudinal constraints, factoring in aerodynamic drag, rolling resistance, a 92% propulsion efficiency, and a 75% regenerative braking efficiency.

**Performance Benchmark**
The system is evaluated using a Two-Car Race benchmark against a highly variable synthetic traffic profile (combining urban stop-and-go with highway cruising). 

| Metric | Standard Reactive Driver | Eco-MPC (Predictive) | Improvement |
| :--- | :--- | :--- | :--- |
| **Driving Style** | Harsh acceleration and braking | Smooth trajectory matching | Optimized Comfort |
| **Safety Gap** | Erratic (frequent overshoot) | Strictly Maintained (15m - 35m) | Zero Violations |
| **Energy Consumed** | ~138 Wh | ~132 Wh | **~4.2% Savings** |

**Repository Structure**
* `src/data_generator.py`: Synthesizes complex, multi-scenario driving profiles for AI training.
* `src/lstm_model.py`: Defines, trains, and exports the neural network architecture.
* `src/mpc_controller.py`: Houses the multi-objective cost function and torque optimization logic.
* `src/vehicle_dynamics.py`: Calculates EV physics, motor drain, and regenerative charging.
* `src/run_simulation.py`: Executes the comparative benchmark and renders the dashboard.
* `models/traffic_lstm.keras`: The compiled, production-ready inference weights.

**Getting Started**
Clone the repository and run the benchmark simulation to generate the performance dashboard (`results.png`).

```bash
git clone [https://github.com/Nishita1652/Ecological-PCC-EV.git](https://github.com/Nishita1652/Ecological-PCC-EV.git)
cd Ecological-PCC-EV
python3 -m pip install matplotlib numpy scipy tensorflow scikit-learn
python3 src/run_simulation.py
