import habitat
from habitat import Challenge 
import os

class Eval(Challenge):
    def __init__(self, eval_remote=False, split_l=-1, split_r=-1):
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
        super().__init__(config_paths)


if __name__ == "__main__":
    os.environ["CHALLENGE_CONFIG_FILE"] = f"configs/mp3d.yaml"
    config = habitat.get_config(os.environ["CHALLENGE_CONFIG_FILE"])
    eval = Eval(
        eval_remote=False,
        split_l=1,
        split_r=11
    )