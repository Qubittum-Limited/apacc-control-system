# APACC Simulation Configuration
simulation:
  scenarios: 10000
  parallel_runs: 4
  random_seed: 42

apacc:
  control_frequency: 100  # Hz
  prediction_horizon: 20
  fuzzy_rules: 50
  
safety_thresholds:
  min_ttc: 2.0  # seconds
  max_lateral_deviation: 0.5  # meters
  max_acceleration: 3.0  # m/s^2
  max_jerk: 2.0  # m/s^3
  max_control_latency: 10.0  # ms

carla:
  host: localhost
  ports: [2000, 2001, 2002, 2003]
  timeout: 10.0

data_export:
  format: csv
  include_raw_data: true
  compression: gzip