import gymnasium as gym
def make_env(env_name, idx, capture_video, run_name):
    """
    Creates a thunk (function with no arguments) that initializes a Gym environment with optional video recording and episode statistics recording.
    Args:
        env_name (str): The environment ID to create (as per Gym registry).
        idx (int): The index of the environment (used to determine if video should be captured).
        capture_video (bool): Whether to capture video for the environment (only for idx == 0).
        run_name (str): The name of the run, used to organize video output directories.
    Returns:
        function: A thunk that, when called, returns the initialized Gym environment.
    Note:
        This function is taken from the CleanRL library.
    """
    
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk