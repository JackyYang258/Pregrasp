from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import math

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)
from mani_skill.utils import common, gym_utils, sapien_utils

from mani_skill.envs.utils.observations import (
    parse_obs_mode_to_struct,
    sensor_data_to_pointcloud,
)

from typing import Optional, Dict, Union
from inference import calculate_graspability_from_point_cloud, initialize_graspnet_model

@register_env("Plate-v1", max_episode_steps=50)
class RobotPlateEnv(BaseEnv):

    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.angle_reward = torch.tensor(0.0)
        self.plate_position_penalty = torch.tensor(0.0)
        self.ee_position_penalty = torch.tensor(0.0)
        self.static_reward = torch.tensor(0.0)
        self.graspability = torch.tensor(0.0)
        self.plate_static_reward = torch.tensor(0.0)
        initialize_graspnet_model()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):

        pose_base = sapien_utils.look_at(eye=[0.2, 0.3, 0.3], target=[0.2, 0, 0.1])
        pose_left = sapien_utils.look_at(eye=[0.2, -0.5, 0.5], target=[0.2, 0, 0.0])
        # pose_base = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        # pose_right = sapien_utils.look_at(eye=[0.2, 0.5, 0.3], target=[0, 0, 0.2])
        # pose_left = sapien_utils.look_at(eye=[0.2, -0.5, 0.3], target=[0, 0, 0.2])
        return [
                CameraConfig("base_camera", pose_base, 128, 128, np.pi / 2, 0.01, 100),
                # CameraConfig("right_camera", pose_right, 128, 128, np.pi / 2, 0.01, 100),
                CameraConfig("left_camera", pose_left, 128, 128, np.pi / 2, 0.01, 100)
                ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.plate = actors.build_from_mesh(
            scene=self.scene,
            name="plate",
            initial_pose=sapien.Pose(p=[0, 0, 0.02]),
        )
        # self.obj = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="cube",
        #     body_type="dynamic",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # the table scene initializes two robots. the first one self.agents[0] is on the left and the second one is on the right
            
            plate_xyz = torch.zeros((b, 3))
            plate_xyz[:, 0] = 0.2
            plate_xyz[:, 2] = 0.004
            theta = torch.tensor(math.radians(0))  
        
            qx = torch.tensor(0)
            qy = -torch.sin(theta / 2)
            qz = torch.tensor(0)
            qw = torch.cos(theta / 2)
            plate_quat = torch.zeros((b, 4))
            plate_quat[:, 0] = qx
            plate_quat[:, 1] = qy
            plate_quat[:, 2] = qz
            plate_quat[:, 3] = qw
            self.plate.set_pose(Pose.create_from_pq(p=plate_xyz, q=plate_quat))
            
            # xyz = torch.zeros((b, 3))
            # # xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # xyz[..., 0] = 0.3
            # xyz[..., 1] = 0.15
            # xyz[..., 2] = self.cube_half_size
            # q = [1, 0, 0, 0]
            # obj_pose = Pose.create_from_pq(p=xyz, q=q)
            # self.obj.set_pose(obj_pose)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        # get position of ee pose
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        obs.update(
            plate_pose=self.plate.pose.raw_pose,
        )
        return obs
        # obs = dict(
        #     is_grasped=info["is_grasped"],
        #     tcp_pose=self.agent.tcp.pose.raw_pose,
        #     goal_pos=self.goal_site.pose.p,
        # )
        # if "state" in self.obs_mode:
        #     obs.update(
        #         obj_pose=self.cube.pose.raw_pose,
        #         tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
        #         obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
        #     )
        # return obs

    def get_pointcloud(self):
        info = self.get_info()
        obs = self._get_obs_with_sensor_data(info)
        obs = sensor_data_to_pointcloud(obs, self._sensors)

        # 获取批次大小
        batch_size = obs["pointcloud"]["xyzw"].shape[0]

        # 定义范围过滤的边界
        x_min, x_max = -0.05, 0.45
        y_min, y_max = -0.20, 0.20
        z_min, z_max = 0.005, 0.20

        # 用于存储所有批次的处理结果
        xyz_camera_list = []
        
        # 再次遍历每个批次，裁剪到最小点云数量
        for i in range(batch_size):
            # 提取当前批次的数据
            xyz = obs["pointcloud"]["xyzw"][i, ..., :3]
            colors = obs["pointcloud"]["rgb"][i]
            # segmentation = obs["pointcloud"]["segmentation"][i]
            cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"][i].to(xyz.device)

            # # 计算与目标点的距离
            # target_point = torch.tensor([0.2, 0.0, 0.004], device=xyz.device)
            # distance_threshold = 0.1
            # # 计算点云中每个点到目标点的距离
            # distance_mask = torch.norm(xyz - target_point, dim=1) < distance_threshold
            range_mask = (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) & \
                        (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) & \
                        (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
            
            # 应用范围掩码过滤点云
            xyz = xyz[range_mask]
            colors_filtered = colors[range_mask]
            
            # adjust number of points
            num_point = 1500
            num_valid = len(xyz)
            if num_valid == 0:
                pass
            elif num_valid >= num_point:
                idxs = torch.randperm(num_valid)[:num_point]
                xyz = xyz[idxs]
            else:
                idxs1 = torch.arange(num_valid, device=xyz.device)
                idxs2 = torch.randint(0, num_valid, (num_point - num_valid,), device=xyz.device)
                idxs = torch.cat([idxs1, idxs2])
                xyz = xyz[idxs]
            
            # create a table
            table_z = 0.0  
            table_density = 0.01  
            x_size = x_max - x_min
            y_size = y_max - y_min
            x_points = int(x_size / table_density) + 1
            y_points = int(y_size / table_density) + 1
            total_table_points = x_points * y_points
            x_coords = torch.linspace(x_min, x_max, x_points, device=xyz.device)
            y_coords = torch.linspace(y_min, y_max, y_points, device=xyz.device)
            x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
            table_x = x_grid.reshape(-1)
            table_y = y_grid.reshape(-1)
            table_z = torch.full((total_table_points,), table_z, device=xyz.device)
            table_xyz = torch.stack([table_x, table_y, table_z], dim=1)
            
            if num_valid == 0:
                xyz = table_xyz
            xyz = torch.cat([xyz, table_xyz], dim=0)

            # 将点云转换为齐次坐标 (N, 4)
            xyz_homogeneous = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)

            # 计算相机坐标系下的点云坐标
            world2cam = torch.inverse(cam2world)
            xyz_camera = torch.matmul(world2cam, xyz_homogeneous.t()).t()[:, :3]

            xyz_camera[:, 2] = -xyz_camera[:, 2]
            
            xyz_camera_list.append(xyz_camera.cpu().numpy())

        if batch_size == 1:
            # 如果只有一个批次，直接返回该批次的点云
            return xyz_camera_list[0]
        return xyz_camera_list
    
    def get_obs(self, info: Optional[Dict] = None):

        if info is None:
            info = self.get_info()
        
        state_dict = self._get_obs_state_dict(info)
        obs = common.flatten_state_dict(state_dict, use_torch=True, device=self.device)

        if isinstance(obs, dict):
            data = dict(agent=obs.pop("agent"), extra=obs.pop("extra"))
            obs["state"] = common.flatten_state_dict(data, use_torch=True, device=self.device)
        return obs
    
    def evaluate(self):
        initial_position = torch.tensor([0.2, 0.0, 0.004], device=self.device)
        initial_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        current_position = self.plate.pose.p
        current_quat = self.plate.pose.q

        def get_pitch(q):
            x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            sin_p = 2 * (w * y - z * x)
            sin_p = torch.clamp(sin_p, -1.0, 1.0)
            return torch.asin(sin_p)

        initial_pitch = get_pitch(initial_quat)
        current_pitch = get_pitch(current_quat)

        angle_diff = torch.abs(current_pitch - initial_pitch)
        max_angle = torch.pi / 4  
        angle_diff = torch.clamp(angle_diff, max=max_angle)
        
        is_plate_tilted = angle_diff > 0.3
        is_plate_static = self.plate.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        xy_initial = initial_position[..., :2]
        xy_current = current_position[..., :2]
        xy_diff_squared = torch.sum((xy_current - xy_initial) ** 2, dim=1)
        is_plate_static = is_plate_static & (xy_diff_squared < 0.05) 
        success = is_plate_static & is_plate_tilted
        
        return {
            "success": success.bool(),
            "angle_reward": self.angle_reward,
            "plate_position_penalty": self.plate_position_penalty,
            "ee_position_penalty": self.ee_position_penalty,
            "static_reward": self.static_reward,
            "graspability": self.graspability,
        }
        # is_obj_placed = (
        #     torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        #     <= self.goal_thresh
        # )
        # is_grasped = self.agent.is_grasping(self.cube)
        # is_robot_static = self.agent.is_static(0.2)
        # return {
        #     "success": is_obj_placed & is_robot_static,
        #     "is_obj_placed": is_obj_placed,
        #     "is_robot_static": is_robot_static,
        #     "is_grasped": is_grasped,
        # }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # inference = False
        # if inference:
        #     point_cloud = self.get_pointcloud()
        #     graspability = calculate_graspability_from_point_cloud(point_cloud)
        #     print(graspability)
        
        initial_position = torch.tensor([0.2, 0.0, 0.004], device=self.device)
        initial_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        current_position = self.plate.pose.p
        current_quat = self.plate.pose.q

        def get_pitch(q):
            x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            sin_p = 2 * (w * y - z * x)
            sin_p = torch.clamp(sin_p, -1.0, 1.0)
            return torch.asin(sin_p)

        initial_pitch = get_pitch(initial_quat)
        current_pitch = get_pitch(current_quat)

        # 计算角度差异并设置上限（例如90度）
        angle_diff = torch.abs(current_pitch - initial_pitch)
        max_angle = torch.pi / 4  
        angle_diff = torch.clamp(angle_diff, max=max_angle)

        self.angle_reward = angle_diff

        xy_initial = initial_position[..., :2]
        xy_current = current_position[..., :2]
        xy_diff_squared = torch.sum((xy_current - xy_initial) ** 2, dim=1)
        self.plate_position_penalty = -xy_diff_squared

        contact_position = torch.tensor([0.07, 0.0, 0.0], device=self.device)
        xy_end_effector = self.agent.tcp.pose.p[..., :3]
        xy_diff_squared = torch.sum((xy_end_effector - contact_position[..., :3]) ** 2, dim=1)  
        self.ee_position_penalty = -xy_diff_squared 
        is_plate_inplace = (xy_diff_squared < 0.05) & (angle_diff < 0.5)

        self.is_plate_tilted = angle_diff > 0.05
        qvel_without_gripper = self.agent.robot.get_qvel()
        qvel_without_gripper = qvel_without_gripper[..., :-2]
        self.static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )
        
        v = torch.linalg.norm(self.plate.linear_velocity, axis=1)
        av = torch.linalg.norm(self.plate.angular_velocity, axis=1)
        self.plate_static_reward = 1 - torch.tanh(v * 10 + av)

        inference = False
        if inference:
            point_cloud = self.get_pointcloud()
            point_cloud = point_cloud[self.is_plate_tilted]
            graspability = calculate_graspability_from_point_cloud(point_cloud)
            self.graspability = torch.tensor(graspability, device=self.device)

        reward = 0.2 * self.plate_position_penalty
        reward += 0.1 * self.ee_position_penalty  
        reward += self.static_reward * self.is_plate_tilted
        # reward += 0.2 * self.plate_static_reward
        
        reward += self.angle_reward * is_plate_inplace
        # reward += self.graspability \
        
        reward[info["success"]] = 2.0

        return reward


    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
