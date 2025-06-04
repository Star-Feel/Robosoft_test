import os
import os.path as osp
import pickle
import sys
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from configs import NUM_DATA, RANDOM_GO_DIR, prune_config


@dataclass
class PruneConfig:
    prune_rate: float


def main():
    script_config = PruneConfig(**prune_config)
    for i in tqdm(range(NUM_DATA)):
        local_random_go_dir = osp.join(RANDOM_GO_DIR, f"{i}")
        os.makedirs(local_random_go_dir, exist_ok=True)

        state_action_path = osp.join(local_random_go_dir, "state_action.pkl")
        if not osp.exists(state_action_path):
            print(f"File not found: {state_action_path}")
            continue

        with open(state_action_path, "rb") as f:
            state_action = pickle.load(f)

        new_state_action = {}
        for key, value in state_action.items():
            total_time_steps = len(value)
            prune_time_steps = int(total_time_steps * script_config.prune_rate)
            new_state_action[key] = value[:prune_time_steps]

        old_state_action_path = osp.join(
            local_random_go_dir, "state_action_old.pkl"
        )
        if osp.exists(old_state_action_path):
            print("Be CAREFULLLY !!!!!!!!")
            print(
                "Old state_action file already exists: "
                f"{old_state_action_path}"
            )
            sys.exit(1)
        os.rename(state_action_path, old_state_action_path)

        with open(state_action_path, "wb") as f:
            pickle.dump(new_state_action, f)


if __name__ == "__main__":
    np.random.seed(0)
    main()
