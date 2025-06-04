import json
import os
import os.path as osp

import tqdm

from configs import FULL_DIR, NUM_DATA, RELEASE_DIR, VISUAL_DIR
from ssim.utils import load_json


def main():
    annotations = []
    for i in tqdm.tqdm(range(NUM_DATA)):
        local_full_dir = os.path.join(FULL_DIR, f"{i}")
        local_visual_dir = os.path.join(VISUAL_DIR, f"{i}")
        local_trajectories_dir = os.path.join(
            RELEASE_DIR, "trajectories", f"{i}"
        )
        os.makedirs(local_trajectories_dir, exist_ok=True)

        info = load_json(os.path.join(local_full_dir, "info.json"))

        if osp.exists(viusal_path := osp.join(local_visual_dir, 'visual')):
            os.system(f"mv {viusal_path} {local_trajectories_dir}")

        for _, path in info.items():
            if not isinstance(path, str):
                continue

            if osp.exists(path):
                os.system(f"cp -r {path} {local_trajectories_dir}")

        annotation = {
            "id": info["id"],
            "target_id": info["target_id"],
            "instruction": info["description"],
        }
        annotations.append(annotation)

        with open(osp.join(RELEASE_DIR, "annotations.json"), "w") as f:
            json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    main()
