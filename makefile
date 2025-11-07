run-sitl:
\tPX4_GZ_MODEL_POSE="0,0,0,0,0,0" make px4_sitl jmavsim

run-agent:
\tpython scripts/run_agent.py --config config.yaml