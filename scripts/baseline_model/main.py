import numpy as np
from stable_baselines3 import PPO

from ssim.envs import (
    NavigationSnakeArguments,
    NavigationSnakeTorqueEnvironmentForGymTrain,
)
from ssim.utils import load_json


def main():
    data_path = "./work_dirs/navigation_data/full/0/info.json"
    work_dir = "./work_dirs/navigation_demo"
    # os.chdir("/data/wjs/wrp/SoftRoboticaSimulator")
    # data_path = "/data/wjs/wrp/SoftRoboticaSimulator/test/full/1/info.json"
    # work_dir = "/data/wjs/wrp/SoftRoboticaSimulator/test"
    info = load_json(data_path)

    info['config'] = info['config'].replace(
        './', '/data/zyw/workshop/attempt/'
    )
    info['state_action'] = info['state_action'].replace(
        './', '/data/zyw/workshop/attempt/'
    )

    config = NavigationSnakeArguments.from_yaml(info["config"])
    env = NavigationSnakeTorqueEnvironmentForGymTrain(config)
    env.set_target_id(info["target_id"])
    state = env.reset()
    # env.set_target(info["target_id"])
    print()

    model = PPO("CnnPolicy", env, verbose=1)

    # 训练模型
    model.learn(total_timesteps=10)

    # 保存模型
    model.save("ppo_random_image_env")

    # 测试模型
    obs = env.reset()
    for _ in range(10):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            print(rewards)
            break


if __name__ == "__main__":
    np.random.seed(0)
    main()
