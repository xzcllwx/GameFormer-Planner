import os
import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
from common_utils import *
from .refinement import RefinementPlanner
from .smoother import MotionNonlinearSmoother
from .occupancy_adapter import occupancy_adpter

from nuplan.planning.simulation.path.path import AbstractPath
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from shapely.geometry import Point
import pandas as pd

MAX_ACC = 4.0
MAX_DEC = 7.0
COMFORT_ACC = 1.0

class TrajectoryPlanner:
    def __init__(self, device='cpu'):
        self.N = int(T/DT)
        self.ts = DT
        self._device = device
        self.planner = RefinementPlanner(device)
    
    def plan(self, ego_state, ego_state_transformed, neighbors_state_transformed, 
             predictions, plan, scores, ref_path, observation):
        # Get the plan from the prediction model
        plan = plan[0].cpu().numpy()

        # Get the plan in the reference path
        if ref_path is not None:
            distance_to_ref = scipy.spatial.distance.cdist(plan[:, :2], ref_path[:, :2])
            i = np.argmin(distance_to_ref, axis=1)
            plan = ref_path[i, :3] # 基于学习方案做速度规划
            s = np.concatenate([[0], i]) * 0.1
            speed = np.diff(s) / DT
        else:
            speed = np.diff(plan[:, :2], axis=0) / DT
            speed = np.linalg.norm(speed, axis=-1)
            speed = np.concatenate([speed, [speed[-1]]])
            
        # Refine planning
        if ref_path is None:
            pass
        else:
            occupancy = occupancy_adpter(predictions[0], scores[0, 1:], neighbors_state_transformed[0], ref_path)
            ego_plan_ds = torch.from_numpy(speed).float().unsqueeze(0).to(self._device)
            ego_plan_s = torch.from_numpy(s).float().unsqueeze(0).to(self._device)
            ego_state_transformed = ego_state_transformed.to(self._device)
            ref_path = torch.from_numpy(ref_path).unsqueeze(0).to(self._device)
            occupancy = torch.from_numpy(occupancy).unsqueeze(0).to(self._device)

            s, speed = self.planner.plan(ego_state_transformed, ego_plan_ds, ego_plan_s, occupancy, ref_path)
            s = s.squeeze(0).cpu().numpy()
            speed = speed.squeeze(0).cpu().numpy()

            # Convert to Cartesian trajectory
            ref_path = ref_path.squeeze(0).cpu().numpy()
            i = (s * 10).astype(np.int32).clip(0, len(ref_path)-1)
            plan = ref_path[i, :3]

        return plan
    
    @staticmethod
    def transform_to_Cartesian_path(path, ref_path):
        frenet_idx = np.array(path[:, 0] * 10, dtype=np.int32)
        frenet_idx = np.clip(frenet_idx, 0, len(ref_path)-1)
        ref_points = ref_path[frenet_idx]
        l = path[frenet_idx, 1]

        cartesian_x = ref_points[:, 0] - l * np.sin(ref_points[:, 2])
        cartesian_y = ref_points[:, 1] + l * np.cos(ref_points[:, 2])
        cartesian_path = np.column_stack([cartesian_x, cartesian_y])

        return cartesian_path


def annotate_occupancy(occupancy, ego_path, red_light_lane):
    ego_path_red_light = scipy.spatial.distance.cdist(ego_path[:, :2], red_light_lane)

    if len(red_light_lane) < 80:
        pass
    else:
        occupancy[np.any(ego_path_red_light < 0.5, axis=-1)] = 1

    return occupancy


def annotate_speed(ref_path, speed_limit):
    speed = np.ones(len(ref_path)) * speed_limit
    
    # get the turning point
    turning_idx = np.argmax(np.abs(ref_path[:, 3]) > 1/10)

    # set speed limit to 3 m/s for turning
    if turning_idx > 0:
        speed[turning_idx:] = 3

    return speed[:, None]


def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi


def transform_to_ego_frame(path, ego_state):
    ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
    path_x, path_y = path[:, 0], path[:, 1]
    ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
    ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (path_y - ego_y)
    ego_path = np.stack([ego_path_x, ego_path_y], axis=-1)

    return ego_path

def angle_add(angle1, angle2):
    """
    使用矢量点乘和叉乘正确实现角度相加
    """
    # 将角度转换为单位向量
    v1 = np.array([np.cos(angle1), np.sin(angle1)])
    v2 = np.array([np.cos(angle2), np.sin(angle2)])
    
    # 使用旋转矩阵实现旋转
    # R = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    # 应用旋转: v2旋转v1
    x_new = v1[0] * v2[0] - v1[1] * v2[1]
    y_new = v1[0] * v2[1] + v1[1] * v2[0]
    
    # 提取角度
    return np.arctan2(y_new, x_new)

def angle_sub(angle1, angle2):
    """
    使用矢量点乘和叉乘正确实现角度相减
    """
    # 将角度转换为单位向量
    v1 = np.array([np.cos(angle1), np.sin(angle1)])
    v2 = np.array([np.cos(angle2), np.sin(angle2)])
    
    # 使用点乘和叉乘计算差角
    dot_product = np.dot(v1, v2)  # |v1|*|v2|*cos(θ)
    cross_product = np.cross(v1, v2)  # |v1|*|v2|*sin(θ)
    
    # 计算角度差
    diff_angle = np.arctan2(cross_product, dot_product)
    
    return diff_angle

def calculate_path_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

    return curvature

def find_intersection_indices(trajectory, intersection_polygon):
    """
    找到交点区域对应轨迹中的索引
    
    参数:
        trajectory: 轨迹点数组，形状为 [D, T] 或 [T, D]
        intersection_polygon: Shapely的几何对象(通常是Polygon)
    
    返回:
        trajectory_indices: 与交点区域相交的轨迹点索引列表
    """
    # 确保trajectory形状正确 [T, 2]
    if trajectory.shape[0] == 2:
        trajectory = trajectory.T  # 转置为 [T, 2]
    
    # 获取交点区域的代表点(通常使用中心点)
    if intersection_polygon.geom_type == 'Polygon':
        representative_point = intersection_polygon.centroid
    else:  # MultiPolygon或其他类型
        # 获取面积最大的部分
        if hasattr(intersection_polygon, 'geoms'):
            largest_poly = max(intersection_polygon.geoms, key=lambda x: x.area)
            representative_point = largest_poly.centroid
        else:
            representative_point = intersection_polygon.centroid
    
    # 计算轨迹点到交点的距离
    distances = []
    for i, point in enumerate(trajectory):
        point_obj = Point(point[0], point[1])
        distances.append((i, point_obj.distance(representative_point)))
    
    # 找到距离最小的点的索引
    closest_idx = min(distances, key=lambda x: x[1])[0]
    
    # 也可以返回多个靠近交点的索引
    threshold_distance = 2.0  # 根据应用调整阈值
    near_indices = [idx for idx, dist in distances if dist < threshold_distance]
    
    return closest_idx, near_indices

def LF_constant_prediction(is_leader, agent_v, target_t_ind, path, target_speed, agent_prediction, T, DT, s_index):
    s = 0
    if target_t_ind == 0:
        target_t_ind += 1
    for t in range(s_index + 1):
        dx = agent_prediction[0, t + 1] - agent_prediction[0, t]
        dy = agent_prediction[1, t + 1] - agent_prediction[1, t]
        s += np.sqrt(dx**2 + dy**2)
    v = np.linalg.norm(agent_v)
    target_t = target_t_ind * DT
    # Calculate acceleration a for agent to reach distance s at time t
    a = 2 * (s - v * target_t) / (target_t) ** 2
    if is_leader:
        a = np.clip(a, 0, MAX_ACC)
    else:
        a = np.clip(a, -MAX_DEC, 0)
    # print(f"Leader Agent acceleration to reach distance {s} at time {target_t}: {a}")
    t_vec_1 = np.arange(1, target_t_ind + 1) * DT
    s_vec_1 = t_vec_1 * v + 0.5 * a * t_vec_1**2
    # Calculate positions based on s_vector
    v_update = v + a * target_t
    if v_update < 0:
        print("reverse case")
        v_update = 0
        a = -MAX_DEC
        target_t_ind = np.ceil(( - v) / a / DT)
        if target_t_ind < 1:
            target_t_ind = 1
        t_vec_1 = np.arange(1, target_t_ind + 1) * DT
        s_vec_1 = t_vec_1 * v + 0.5 * a * t_vec_1**2
    v_target = target_speed
    if v_update > v_target:
        a = -COMFORT_ACC
    else:
        a = COMFORT_ACC
    target_t_2_ind = np.ceil((v_target - v_update) / a /DT)
    add_constant = False
    if target_t_ind + target_t_2_ind >= T * FREQUENCE:
        target_t_2_ind = T * FREQUENCE - target_t_ind
    else:
        add_constant = True
    target_t_2 = target_t_2_ind * DT
    t_vec_2 = np.arange(1, target_t_2_ind + 1) * DT
    s_vec_2 = t_vec_2 * v_update + 0.5 * a * t_vec_2 ** 2 + s_vec_1[-1]
    if add_constant:
        t_vec_3 = np.arange(1, (T*FREQUENCE - target_t_ind -target_t_2_ind) + 1) * DT
        s_vec_3 = t_vec_3 * v_target + s_vec_2[-1]
        s_vec = np.concatenate([s_vec_1, s_vec_2, s_vec_3])
    else:
        s_vec = np.concatenate([s_vec_1, s_vec_2])

    pred_x = []
    pred_y = []
    for i, s in enumerate(s_vec): 
        x, y = path.calc_position(s)
        if (len(pred_x) > 0):
            dx = x - pred_x[-1]
            dy = y - pred_y[-1]
            distance = np.sqrt(dx**2 + dy**2)
            if distance > s_vec[i]-s_vec[i-1] + 0.5:
                # print(f"Distance between two points: {distance} > {s_vec[i]-s_vec[i-1]+0.5}")
                heading = np.arctan2(agent_prediction[1, -1] - agent_prediction[1, -5], 
                                     agent_prediction[0, -1] - agent_prediction[0, -5])
                pred_x.append(pred_x[-1] + (s_vec[i]-s_vec[i-1]) * np.cos(heading))
                pred_y.append(pred_y[-1] + (s_vec[i]-s_vec[i-1]) * np.sin(heading))
                continue
        pred_x.append(x)
        pred_y.append(y)
    pred_x = np.array(pred_x)
    pred_y = np.array(pred_y)
    return pred_x, pred_y

def calcualte_branch_time(l_b, l_pred, f_b, f_pred, ego_v, tangent_vector, history_branch_time):
    sigma_leader = 2.0
    sigma_follower = 2.0
    rho = 0.25
    # leader branch 
    # l_b的维度为[2, T+1]， 差分计算各个点的速度[2, T]
    l_agent_traj = np.transpose(l_pred, (1, 0))
    l_agent_v = np.diff(l_agent_traj, axis=0) / DT
    f_agent_traj = np.transpose(f_pred, (1, 0))
    f_agent_v = np.diff(f_agent_traj, axis=0) / DT
    l_relative_v = l_agent_v - ego_v
    f_relative_v = f_agent_v - ego_v
    l_pred_v = np.dot(l_relative_v, tangent_vector)
    f_pred_v = np.dot(f_relative_v, tangent_vector)
    diff = l_pred_v - l_pred_v
    # 正太分布根据diff计算观测概率
    p_leader_ob = 1 / (np.sqrt(2 * np.pi) * sigma_leader) * np.exp(-(diff)**2 / (2 * sigma_leader**2)) 

    diff = l_pred_v - f_pred_v
    p_follower_ob = 1 / (np.sqrt(2 * np.pi) * sigma_follower) * np.exp(-(diff)**2 / (2 * sigma_follower**2))
    k_ob_l_f = p_follower_ob / (p_leader_ob + p_follower_ob)
    k_ob_f_l = p_leader_ob / (p_leader_ob + p_follower_ob)
    # calculate belief
    p_leader = p_leader_ob * l_b + k_ob_f_l * f_b * 1 / (np.sqrt(2 * np.pi) * sigma_follower) * rho
    p_follower = p_follower_ob * f_b + k_ob_l_f * l_b * 1 / (np.sqrt(2 * np.pi) * sigma_follower) * rho
    sum_p = p_leader + p_follower
    p_leader = p_leader / sum_p
    p_follower = p_follower / sum_p
    # print(f"leader_p_leader_ob: {p_leader}")
    # Calculate the entropy of l_b
    leader_entropy = -np.cumsum(p_leader * np.log2(p_leader + 1e-9), axis=0) - np.cumsum(p_follower * np.log2(p_follower + 1e-9), axis=0)
    
    diff = f_pred_v - l_pred_v
    # 正太分布根据diff计算观测概率
    p_leader_ob = 1 / (np.sqrt(2 * np.pi) * sigma_leader) * np.exp(-(diff)**2 / (2 * sigma_leader**2)) 
    diff = f_pred_v - f_pred_v
    p_follower_ob = 1 / (np.sqrt(2 * np.pi) * sigma_follower) * np.exp(-(diff)**2 / (2 * sigma_follower**2))
    k_ob_l_f = p_follower_ob / (p_leader_ob + p_follower_ob)
    k_ob_f_l = p_leader_ob / (p_leader_ob + p_follower_ob)
    # calculate belief
    p_leader = p_leader_ob * l_b + k_ob_f_l * f_b * 1 / (np.sqrt(2 * np.pi) * sigma_follower) * rho
    p_follower = p_follower_ob * f_b + k_ob_l_f * l_b * 1 / (np.sqrt(2 * np.pi) * sigma_follower) * rho
    sum_p = p_leader + p_follower
    p_leader = p_leader / sum_p
    p_follower = p_follower / sum_p
    # print(f"follower_p_leader_ob: {p_follower}")
    # Calculate the entropy of l_b
    follower_entropy = -np.cumsum(p_follower * np.log2(p_follower + 1e-9), axis=0) - np.cumsum(p_leader * np.log2(p_leader + 1e-9), axis=0)
    # print(f"leader_entropy: {leader_entropy}")
    # print(f"follower_entropy: {follower_entropy}")
    
    # 计算分支时间
    # Find the first index where leader_entropy exceeds a specified threshold
    threshold_value = 1 + 4.5 # You can adjust this threshold as needed
    alpha = 0.2
    leader_branch_time_index = np.argmax(leader_entropy > threshold_value)
    # print(f"Branch time index where leader_entropy exceeds {threshold_value}: {leader_branch_time_index}")
    if leader_branch_time_index == 0:
        leader_branch_time_index = T * FREQUENCE - 10
    follower_branch_time_index = np.argmax(follower_entropy > threshold_value)
    # print(f"Branch time index where follower_entropy exceeds {threshold_value}: {follower_branch_time_index}")
    if follower_branch_time_index == 0:
        follower_branch_time_index = T * FREQUENCE - 10
    branch_time = min(leader_branch_time_index, follower_branch_time_index)
    # branch_time = np.ceil((leader_branch_time_index+follower_branch_time_index)/2)
    print(f"Branch time: {branch_time}")
    if len(history_branch_time) > 0:
        branch_time = int(history_branch_time[-1] + (branch_time - history_branch_time[-1]) * 0.2)
    branch_time = np.clip(branch_time, 5, T * FREQUENCE - 10)
    print(f"Branch time after clip: {branch_time}")
    
    return branch_time

def save_data_to_csv(frame_data: dict, iteration, csv_save_dir):
    # 创建目录(如果不存在)
    os.makedirs(os.path.dirname(csv_save_dir), exist_ok=True)
    df = pd.DataFrame([frame_data])
    csv_file = os.path.join(csv_save_dir, f"{iteration}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")