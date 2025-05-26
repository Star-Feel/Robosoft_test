import os
import os.path as osp
import json

import tqdm
from ssim.utils import load_json

RELEASE_DIR = "work_dirs/navigation_data/release"
VISUAL_DIR = "work_dirs/navigation_data/visual"
FULL_DIR = "work_dirs/navigation_data/full"

N = 10


def main():
    annotations = []
    for i in tqdm.tqdm(range(N)):
        local_full_dir = os.path.join(FULL_DIR, f"{i}")
        local_visual_dir = os.path.join(VISUAL_DIR, f"{i}")
        local_release_dir = os.path.join(RELEASE_DIR, f"{i}")
        os.makedirs(local_release_dir, exist_ok=True)

        info = load_json(os.path.join(local_full_dir, "info.json"))

        if osp.exists(viusal_path := osp.join(local_visual_dir, 'visual')):
            os.system(f"mv {viusal_path} {local_release_dir}")

        for _, path in info.items():
            if not isinstance(path, str):
                continue

            if osp.exists(path):
                os.system(f"cp -r {path} {local_release_dir}")

        annotation = {
            "id": info["id"],
            "target_id": info["target_id"],
            "description": info["description"],
        }
        annotations.append(annotation)

        with open(osp.join(RELEASE_DIR, "annotations.json"), "w") as f:
            json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    main()
