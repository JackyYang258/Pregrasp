import os
import sys
import numpy as np
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = "/home/ysq/project/graspness_unofficial"
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector

import MinkowskiEngine as ME
import collections.abc as container_abcs

_GRASPNET_MODEL = None
_DEVICE = None

def initialize_graspnet_model(device="cuda:0"):
    """初始化模型并缓存"""
    global _GRASPNET_MODEL, _DEVICE
    checkpoint_path = '/home/ysq/project/graspness_unofficial/logs/minkresunet_epoch10-1.tar'
    if torch.cuda.is_available() and "cuda" in device:
        _DEVICE = torch.device(device)
    else:
        _DEVICE = torch.device("cpu")
    
    if _GRASPNET_MODEL is None:
        # 初始化模型
        net = GraspNet(seed_feat_dim=512, is_training=False)
        net.to(_DEVICE)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=_DEVICE)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        
        _GRASPNET_MODEL = net
    print

def calculate_graspability_from_point_cloud(
    point_clouds,
    voxel_size=0.005,
    collision_thresh=-1,
    voxel_size_cd=0.01,
    visualize=False
):
    global _GRASPNET_MODEL, _DEVICE

    # 数据预处理
    ret_dicts = []
    for pc in point_clouds:
        ret_dict = {
            'point_clouds': pc.astype(np.float32),
            'coors': pc.astype(np.float32) / voxel_size,
            'feats': np.ones_like(pc).astype(np.float32),
        }
        ret_dicts.append(ret_dict)
    
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in ret_dicts],
                                                                [d["feats"] for d in ret_dicts])
    #todo:test contiguousarray
    coordinates_batch = np.ascontiguousarray(coordinates_batch, dtype=np.int32)
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res

    batch_data = collate_fn_(ret_dicts)

    for key in batch_data:
        if isinstance(batch_data[key], list):
            for i in range(len(batch_data[key])):
                batch_data[key][i] = batch_data[key][i].to(_DEVICE)
        else:
            batch_data[key] = batch_data[key].to(_DEVICE)

    with torch.no_grad():
        # 模型推理
        end_points = _GRASPNET_MODEL(batch_data)
        grasp_preds = pred_decode(end_points)
    
    graspabilities = []
    for i in range(len(point_clouds)):
        preds = grasp_preds[i].detach().cpu().numpy()
        # Filtering grasp poses for real-world execution. 
        # The first mask preserves the grasp poses that are within a 30-degree angle with the vertical pose and have a width of less than 9cm.
        mask = (preds[:,10] > 0.9) & (preds[:,1] < 0.09)
        # preds = preds[mask]

        gg = GraspGroup(preds)
        # 碰撞检测
        if collision_thresh > 0:
            mfcdetector = ModelFreeCollisionDetector(point_clouds[i], voxel_size=voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
            # gg = gg[~collision_mask]
        
        gg = gg.nms()
        gg = gg.sort_by_score()
        if gg.__len__() > 30:
            gg = gg[:30]

        # 计算 graspability
        gg = gg[gg.scores >= 0.1]
        graspability = np.sum(gg.scores)
        graspabilities.append(graspability)
        
        vis = True
        if vis:
            pc = ret_dicts[0]['point_clouds']
            grippers = gg.to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
            o3d.visualization.draw_geometries([cloud, *grippers])
        
        # visualization for debugging
        # if i == 1:
        #     grippers = gg.to_open3d_geometry_list()
        #     cloud = o3d.geometry.PointCloud()
        #     cloud.points = o3d.utility.Vector3dVector(point_clouds[i].astype(np.float32))
        #     o3d.visualization.draw_geometries([cloud, *grippers])
    
    return graspabilities


# 示例用法
if __name__ == '__main__':
    # 加载点云数据
    point_clouds = [
        np.load("/home/ysq/project/maniskill/pointcloud/point_cloud_vertical.npy"),
        np.load("/home/ysq/project/maniskill/pointcloud/point_cloud.npy")
    ]
    for point_cloud in point_clouds:
        point_cloud[:, 2] = -point_cloud[:, 2]


    # 计算 graspability
    graspabilities = calculate_graspability_from_point_cloud(point_clouds)
    for i, graspability in enumerate(graspabilities):
        print(f'Graspability for point cloud {i}: {graspability}')