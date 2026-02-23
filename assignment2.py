import gymnasium as gym
import numpy as np


def load_environment(render_mode=None):
    """Load and return the Taxi-v3 Gymnasium environment."""
    env = gym.make("Taxi-v3", render_mode=render_mode)
    return env


def print_env_info(env):
    """Print basic information about the environment."""
    print("=" * 50)
    print("Taxi-v3 Environment Info")
    print("=" * 50)
    print(f"Action Space:       {env.action_space}")
    print(f"Observation Space:  {env.observation_space}")
    print(f"Number of States:   {env.observation_space.n}")
    print(f"Number of Actions:  {env.action_space.n}")
    print()
    action_meanings = {
        0: "Move South (down)",
        1: "Move North (up)",
        2: "Move East (right)",
        3: "Move West (left)",
        4: "Pickup passenger",
        5: "Drop off passenger",
    }
    print("Action Meanings:")
    for a, desc in action_meanings.items():
        print(f"  {a}: {desc}")
    print()


def decode_state(state: int):
    """
    Decode a state scalar into its components.
    State encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination

    Parameters
    ----------
    state : int
        Encoded state value (0 – 499)

    Returns
    -------
    dict with taxi_row, taxi_col, passenger_loc, destination
    """
    destination = state % 4
    state //= 4
    passenger_loc = state % 5
    state //= 5
    taxi_col = state % 5
    taxi_row = state // 5

    passenger_map = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue", 4: "In Taxi"}
    destination_map = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue"}

    info = {
        "taxi_row": taxi_row,
        "taxi_col": taxi_col,
        "passenger_loc": passenger_map[passenger_loc],
        "destination": destination_map[destination],
    }
    return info


def print_state_info(state: int):
    """Print a human-readable description of a state scalar."""
    info = decode_state(state)
    print(f"State {state}:")
    print(f"  Taxi Position  : row={info['taxi_row']}, col={info['taxi_col']}")
    print(f"  Passenger Loc  : {info['passenger_loc']}")
    print(f"  Destination    : {info['destination']}")
    print()


if __name__ == "__main__":
    env = load_environment()
    print_env_info(env)
    obs, _ = env.reset()
    print_state_info(obs)
    env.close()
