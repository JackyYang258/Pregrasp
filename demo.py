import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Actor, Link
import torch


import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

import trimesh
import trimesh.scene
from inference import calculate_graspability_from_point_cloud, initialize_graspnet_model
@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""
    
    cam_width: Optional[int] = None
    
    cam_height: Optional[int] = None

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""

def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    if args.render_mode == "None":
        args.render_mode = None
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    sensor_configs["shader_pack"] = args.shader
    
    print("obs_mode", args.obs_mode)
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=sensor_configs,
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    env: BaseEnv = gym.make(
        args.env_id,
        **env_kwargs
    )
    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, info_on_video=False, save_trajectory=False, max_steps_per_video=env._max_episode_steps)

    if verbose:
        # print("Observation space", env.observation_space)
        # print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)

    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode is not None:
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    while True:
        action = env.action_space.sample() if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(None)
        obs = env.get_pointcloud()
        # for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
            # if isinstance(obj, Actor):
            #     print(f"{obj_id}: Actor, name - {obj.name}")
            # elif isinstance(obj, Link):
            #     print(f"{obj_id}: Link, name - {obj.name}")
        if args.obs_mode == "pointcloud":
            # xyz = obs["pointcloud"]["xyzw"][0, ..., :3]  # 点云坐标 (N, 3)
            # colors = obs["pointcloud"]["rgb"][0]  # 点云颜色 (N, 3)
            # segmentation = obs["pointcloud"]["segmentation"][0]  # 分割标签 (N,)
            # cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"][0]  # 相机到世界的变换矩阵 (4, 4)

            # # 定义范围过滤的边界
            # x_min, x_max = -0.3, 0.5
            # y_min, y_max = -0.5, 0.5
            # z_min, z_max = -1, 0.2

            # # 创建范围掩码
            # range_mask = (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) & \
            #             (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) & \
            #             (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)

            # # 应用范围掩码过滤点云
            # xyz_filtered = xyz[range_mask]
            # colors_filtered = colors[range_mask]

            # # 将 cam2world 转换为 Tensor
            # cam2world_tensor = torch.tensor(cam2world, device=xyz.device, dtype=xyz.dtype)

            # # 将点云转换为齐次坐标 (N, 4)
            # xyz_homogeneous = torch.cat([xyz_filtered, torch.ones(xyz_filtered.shape[0], 1, device=xyz.device) ], dim=1)

            # # 计算相机坐标系下的点云坐标
            # world2cam = torch.inverse(cam2world_tensor)
            # xyz_camera = torch.matmul(world2cam, xyz_homogeneous.t()).t()[:, :3]

            # # 缩放点云坐标
            # xyz_camera = xyz_camera
            point_cloud = env.get_pointcloud()

            # 将结果保存为 NumPy 数组
            xyz_camera_np = point_cloud
            np.save('pointcloud/point_cloud_vertical1.npy', xyz_camera_np)
            initialize_graspnet_model()
            graspability = calculate_graspability_from_point_cloud(point_clouds=[xyz_camera_np])
            print(graspability)
            
            # 使用 trimesh 可视化点云
            pcd = trimesh.points.PointCloud(xyz_camera_np)
            for uid, config in env.unwrapped._sensor_configs.items():
                if isinstance(config, CameraConfig):
                    cam2world = np.eye(4)
                    camera = trimesh.scene.Camera(uid, (1024, 1024), fov=(np.rad2deg(config.fov), np.rad2deg(config.fov)))
                    break
            trimesh.Scene([pcd], camera=camera, camera_transform=cam2world).show()
        # if args.obs_mode == "pointcloud":
        #     xyz = obs["pointcloud"]["xyzw"][0, ..., :3]
        #     colors = obs["pointcloud"]["rgb"][0]
        #     xyz = xyz.cpu().numpy()
        #     colors = colors.cpu().numpy()       
        #     segmentation = obs["pointcloud"]["segmentation"][0]  # 假设分割标签存储在这里
        #     cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"][0]

        #     # 将 segmentation 从 GPU 复制到 CPU 并转换为 NumPy 数组
        #     segmentation = segmentation.cpu().numpy()

        #     # target_segmentation_values = [32] + list(range(1, 31))  # 1 到 20 的标签值

        #     # mask = ~np.isin(segmentation[:, 0], target_segmentation_values)

        #     # # 应用掩码过滤点云
        #     # xyz_filtered = xyz[mask]
        #     # colors_filtered = colors[mask]
            
        #     x_min, x_max = -0.3, 0.5
        #     y_min, y_max = -0.5, 0.5
        #     z_min, z_max = -1, 1

        #     # 创建一个布尔掩码，表示每个点是否在指定的范围内
        #     xyz_filtered = xyz
        #     colors_filtered = colors
        #     range_mask = (xyz_filtered[:, 0] >= x_min) & (xyz_filtered[:, 0] <= x_max) & \
        #                 (xyz_filtered[:, 1] >= y_min) & (xyz_filtered[:, 1] <= y_max) & \
        #                 (xyz_filtered[:, 2] >= z_min) & (xyz_filtered[:, 2] <= z_max)

        #     # 应用范围掩码过滤点云
        #     xyz_filtered = xyz_filtered[range_mask]
        #     colors_filtered = colors_filtered[range_mask]
            
        #     world2cam = np.linalg.inv(cam2world)
        #     xyz_homogeneous = np.hstack((xyz_filtered, np.ones((xyz_filtered.shape[0], 1))))
        #     xyz_camera = np.dot(world2cam, xyz_homogeneous.T).T
        #     xyz_camera = xyz_camera[:, :3]
            
        #     # print(xyz_camera)
        #     xyz_camera = xyz_camera / 2
        #     np.save('pointcloud/point_cloud_vertical.npy', xyz_camera)

        #     # pcd = trimesh.points.PointCloud(xyz_filtered, colors_filtered)
        #     # for uid, config in env.unwrapped._sensor_configs.items():
        #     #     if isinstance(config, CameraConfig):
        #     #         cam2world = obs["sensor_param"][uid]["cam2world_gl"][0]
        #     #         camera = trimesh.scene.Camera(uid, (1024, 1024), fov=(np.rad2deg(config.fov), np.rad2deg(config.fov)))
        #     #         break
        #     # trimesh.Scene([pcd], camera=camera, camera_transform=cam2world).show()
        #     pcd = trimesh.points.PointCloud(xyz_camera, colors_filtered)
        #     for uid, config in env.unwrapped._sensor_configs.items():
        #         if isinstance(config, CameraConfig):
        #             cam2world = np.eye(4)
        #             camera = trimesh.scene.Camera(uid, (1024, 1024), fov=(np.rad2deg(config.fov), np.rad2deg(config.fov)))
        #             break
        #     trimesh.Scene([pcd], camera=camera, camera_transform=cam2world).show()
        
        # if verbose:
        #     print("reward", reward)
        #     print("terminated", terminated)
        #     print("truncated", truncated)
        #     print("info", info)
        if args.render_mode is not None:
            env.render()
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    env.close()

    if record_dir:
        print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
