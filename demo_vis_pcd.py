import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Actor, Link
import trimesh
import trimesh.scene
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--cam-width", type=int, help="Override the width of every camera in the environment")
    parser.add_argument("--cam-height", type=int, help="Override the height of every camera in the environment")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        reward_mode="none",
        sensor_configs=sensor_configs,
    )

    obs, _ = env.reset(seed=args.seed)
    for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
        if isinstance(obj, Actor):
            print(f"{obj_id}: Actor, name - {obj.name}")
        elif isinstance(obj, Link):
            print(f"{obj_id}: Link, name - {obj.name}")
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        xyz = obs["pointcloud"]["xyzw"][0, ..., :3]
        colors = obs["pointcloud"]["rgb"][0]
        
        segmentation = obs["pointcloud"]["segmentation"][0] 

        target_segmentation_values = [17] + list(range(1, 16)) 

        mask = ~np.isin(segmentation[:, 0], target_segmentation_values)
        xyz = xyz[mask]
        colors = colors[mask]
        np.save('/home/ysq/project/maniskill/point_cloud_cube.npy', xyz)
        pcd = trimesh.points.PointCloud(xyz, colors)
        
        


        # view from first camera
        for uid, config in env.unwrapped._sensor_configs.items():
            if isinstance(config, CameraConfig):
                cam2world = obs["sensor_param"][uid]["cam2world_gl"][0]
                camera = trimesh.scene.Camera(uid, (1024, 1024), fov=(np.rad2deg(config.fov), np.rad2deg(config.fov)))
            break
        trimesh.Scene([pcd], camera=camera, camera_transform=cam2world).show()
        if terminated or truncated:
            break
    env.close()

if __name__ == "__main__":
    main(parse_args())
