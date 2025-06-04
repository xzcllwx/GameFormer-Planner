# csv 格式
# Time(s),X,Y,Z,Yaw,Velocity,Acc,AngularRate
# 5.3116490840911865,343.72674560546875,-252.6216583251953,33.38032531738281,0.46962069478110957,1.475578833182253,-0.15404579325135592,3.4906585039886596e-06
import sys
import os
from nuplan.common.actor_state.ego_state import EgoState
sys.path.append("/root/xzcllwx_ws")
sys.path.append("/root/xzcllwx_ws/GameFormer-Planner")
from parse_xodr.parse_and_visualize import *
from pluto.src.feature_builders.nuplan_scenario_render import NuplanScenarioRender
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from GameFormer.predictor import GameFormer
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.ego_state import EgoState
DT = 0.1
T = 80
H = 21
Offset = 80
W = 1.9
L = 4.5

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

class SceneLoader(NuplanScenarioRender):
    def __init__(self, device='cpu', ego_csv_file=None, npc_csv_file=None):
        super().__init__()  # 调用父类构造函数
        self.N = int(T/DT)
        self.ts = DT
        self._device = device
        self.ego_csv = pd.read_csv(ego_csv_file)
        self.npc_csv = pd.read_csv(npc_csv_file)
        self.ego_data = {}
        self.npc_data = {}
        time = self.ego_csv['Time(s)']
        x = self.ego_csv['X']
        y = self.ego_csv['Y']
        heading = self.ego_csv['Yaw']
        velocity = self.ego_csv['Velocity']
        acceleration = self.ego_csv['Acc']
        augular_rate = self.ego_csv['AngularRate']
        npc_x = self.npc_csv['X']
        npc_y = self.npc_csv['Y']
        npc_heading = self.npc_csv['Yaw']
        npc_velocity = self.npc_csv['Velocity']

        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)

        # Find the index of the first point with a distance greater than a threshold
        threshold = 5  # Adjust the threshold as needed
        first_index = np.argmax(distances > threshold) + 5
        print(first_index)
        dx = np.diff(npc_x)
        dy = np.diff(npc_y)
        distances_npc = np.sqrt(dx**2 + dy**2)
        # Find the index of the first point with a distance greater than a threshold
        threshold_npc = 50  # Adjust the threshold as needed
        first_index_npc = np.argmax(distances_npc > threshold_npc)
        print(first_index_npc)

        first_index = max(first_index, first_index_npc)
        x = x.iloc[first_index:]
        y = y.iloc[first_index:]
        velocity = velocity.iloc[first_index:]
        acceleration = acceleration.iloc[first_index:]
        augular_rate = augular_rate.iloc[first_index:]
        heading = heading.iloc[first_index:]
        time = time.iloc[first_index:]
        npc_x = npc_x.iloc[first_index:]
        npc_y = npc_y.iloc[first_index:]
        npc_velocity = npc_velocity.iloc[first_index:]
        npc_heading = npc_heading.iloc[first_index:]

        # Filter and interpolate the data
        time_filtered = np.arange(time.iloc[0], time.iloc[-1], 0.1)
        x_filtered = np.interp(time_filtered, time, x)
        y_filtered = np.interp(time_filtered, time, y)
        heading_filtered = np.interp(time_filtered, time, heading)
        # velocity_filtered = np.interp(time_filtered, time, velocity)
        acceleration_filtered = np.interp(time_filtered, time, acceleration)
        augular_rate_filtered = np.interp(time_filtered, time, augular_rate)

        npc_x_filtered = np.interp(time_filtered, time, npc_x)
        npc_y_filtered = np.interp(time_filtered, time, npc_y)
        npc_velocity_filtered = np.interp(time_filtered, time, npc_velocity)
        npc_heading_filtered = np.interp(time_filtered, time, npc_heading)
        
        window_size = 3
        self.time = time_filtered[:-window_size+1]
        x_smoothed_data = np.convolve(x_filtered, np.ones(window_size)/window_size, mode='valid')
        y_smoothed_data = np.convolve(y_filtered, np.ones(window_size)/window_size, mode='valid')
        heading_smoothed_data = np.convolve(heading_filtered, np.ones(window_size)/window_size, mode='valid')
        acceleration_smoothed_data = np.convolve(acceleration_filtered, np.ones(window_size)/window_size, mode='valid')
        augular_rate_smoothed = np.convolve(augular_rate_filtered, np.ones(window_size)/window_size, mode='valid')
        x = np.array(x_smoothed_data)
        y = np.array(y_smoothed_data)
        dx = np.diff(x)
        dy = np.diff(y)
        dt = 0.1
        vx = dx / dt
        ax = np.diff(vx) / dt
        ax = np.concatenate(([0], [0], ax))
        print(x.shape)
        vx = np.concatenate(([0], vx))  # Add initial velocity
        print(vx.shape)
        print(ax.shape)
        vy = dy / dt
        ay = np.diff(vy) / dt
        vy = np.concatenate(([0], vy))  # Add initial velocity
        velocity_smoothed = np.sqrt(dx**2 + dy**2) / dt
        velocity_smoothed = np.concatenate(([0], velocity_smoothed))  # Add initial velocity
        wheelbase = 2.65
        steering_angle = np.arctan(augular_rate_smoothed * wheelbase / (velocity_smoothed + 0.001))
        self.ego_data['x'] = x_smoothed_data
        self.ego_data['y'] = y_smoothed_data
        self.ego_data['heading'] = heading_smoothed_data
        self.ego_data['velocity'] = velocity_smoothed
        self.ego_data['acceleration'] = acceleration_smoothed_data
        self.ego_data['steering_angle'] = steering_angle
        self.ego_data['angular_rate'] = augular_rate_smoothed
        self.ego_data['vx'] = vx
        self.ego_data['vy'] = vy
        self.ego_data['ax'] = ax
        self.ego_data['ay'] = ay
        
        self.route = {}
        self.route['x'] = x_smoothed_data
        self.route['y'] = y_smoothed_data
        self.route['heading'] = heading_smoothed_data
        self.route['velocity'] = velocity_smoothed
    
        # check direction
        direction_vector = None
        dir_length = 5
        ind_length = 1
        for i in range(len(self.route['y']) - ind_length):
            if i % dir_length == 0 and i + dir_length < len(self.route['y']):
                direction_vector = np.array([self.route['x'][i + dir_length] - self.route['x'][i], self.route['y'][i + dir_length] - self.route['y'][i]])
                direction_vector /= np.linalg.norm(direction_vector)
            vector_1 = np.array([self.route['x'][i + ind_length] - self.route['x'][i], self.route['y'][i + ind_length] - self.route['y'][i]])
            dir = np.dot(vector_1, direction_vector)
            if dir < 0:
                print(f"Direction changed at index {i + 1}")

        
        npc_x_smoothed_data = np.convolve(npc_x_filtered, np.ones(window_size)/window_size, mode='valid')
        npc_y_smoothed_data = np.convolve(npc_y_filtered, np.ones(window_size)/window_size, mode='valid')
        npc_heading_smoothed_data = np.convolve(npc_heading_filtered, np.ones(window_size)/window_size, mode='valid')
        x = np.array(npc_x_smoothed_data)
        y = np.array(npc_y_smoothed_data)
        dx = np.diff(x)
        dy = np.diff(y)
        dt = 0.1
        vx = dx / dt
        vx = np.concatenate(([0], vx))  # Add initial velocity
        vy = dy / dt
        vy = np.concatenate(([0], vy))  # Add initial velocity
        npc_velocity_smoothed = np.sqrt(dx**2 + dy**2) / dt
        npc_velocity_smoothed = np.concatenate(([0], npc_velocity_smoothed))  # Add initial velocity
        # npc_velocity_smoothed = np.convolve(npc_velocity_filtered, np.ones(window_size)/window_size, mode='valid')
        self.npc_data['x'] = npc_x_smoothed_data
        self.npc_data['y'] = npc_y_smoothed_data
        self.npc_data['velocity'] = npc_velocity_smoothed
        self.npc_data['heading'] = npc_heading_smoothed_data
        yaw_rate = np.diff(npc_heading_smoothed_data) / dt
        yaw_rate = np.concatenate(([0], yaw_rate))
        self.npc_data['yaw_rate'] = yaw_rate
        self.npc_data['vx'] = vx
        self.npc_data['vy'] = vy
        
        self.lanes = {}
        lane_count = 0
        total_areas = get_lane_points_by_id(file=XODR_FILE, step=0.1)
        for k, v in total_areas.items():
            left_lanes_area = v["left_lanes_area"]
            right_lanes_area = v["right_lanes_area"]
            
            for left_lane_id, left_lane_area in left_lanes_area.items():
                self.lanes[lane_count] = {}
                inner_points = left_lane_area["inner"]
                outer_points = left_lane_area["outer"]
                center_points = left_lane_area["center"]
                left_boundary_points = np.array([[x,-y] for x,y in inner_points])
                right_boundary_points = np.array([[x,-y] for x,y in outer_points])
                center_points = np.array([[x,-y] for x,y in center_points]) 
                self.lanes[lane_count]["left_boundary"] = left_boundary_points
                self.lanes[lane_count]["right_boundary"] = right_boundary_points
                self.lanes[lane_count]["center"] = center_points
            lane_count += 1

            for right_lane_id, right_lane_area in right_lanes_area.items():
                self.lanes[lane_count] = {}
                inner_points = right_lane_area["inner"]
                outer_points = right_lane_area["outer"]
                center_points = right_lane_area["center"]
                left_boundary_points = np.array([[x,-y] for x,y in inner_points])
                right_boundary_points = np.array([[x,-y] for x,y in outer_points])
                center_points = np.array([[x,-y] for x,y in center_points])
                self.lanes[lane_count]["left_boundary"] = left_boundary_points
                self.lanes[lane_count]["right_boundary"] = right_boundary_points
                self.lanes[lane_count]["center"] = center_points
            lane_count += 1
            
            self._history_trajectory = []
            self.iteration = None
        
    def plot_data(self):
        # Plot the trajectory
        plt.figure(1)
        plt.plot(self.ego_data['x'], self.ego_data['y'], 'b')
        plt.plot(self.npc_data['x'], self.npc_data['y'], 'r')
        for lane in self.lanes.values():
            left_boundary = lane["left_boundary"]
            right_boundary = lane["right_boundary"]
            center_points = lane["center"]
            plt.plot(left_boundary[:,0], left_boundary[:,1], 'g-')
            plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-')
            plt.plot(center_points[:,0], center_points[:,1], 'g--')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.title('Trajectory')

        # Plot the velocity
        plt.figure(2)
        plt.plot(self.time, self.ego_data['velocity'], 'b')
        plt.plot(self.time, self.npc_data['velocity'], 'g')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity')
        plt.title('Velocity')

        # Plot the acceleration
        plt.figure(3)
        plt.plot(self.time[80:], self.ego_data['acceleration'][80:])
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.title('Acceleration')

        # PLot tire angle
        plt.figure(4)
        plt.plot(self.time, self.ego_data['steering_angle'])
        npc_steer = np.arctan(self.npc_data['yaw_rate'] * 2.65 / (self.npc_data['velocity'] + 0.001))
        plt.plot(self.time, npc_steer)
        
        # Plot yaw
        plt.figure(5)
        plt.plot(self.time, self.ego_data['heading'], 'b-')
        plt.plot(self.time, self.npc_data['heading'], 'r-')
        
        # Plot slip angle
        plt.figure(6)
        plt.plot(self.time, self.ego_data['steering_angle'], 'b-')
        
        plt.show()
    
    def get_initial_state(self):
        t_current = 0 + Offset
        # 提取ego当前状态
        init_state_args = dict()
        init_state_args['position'] = np.array([self.ego_data['x'][t_current], self.ego_data['y'][t_current]])
        init_state_args['orientation'] = self.ego_data['heading'][t_current]
        init_state_args['velocity'] = self.ego_data['velocity'][t_current]
        init_state_args['acceleration'] = self.ego_data['acceleration'][t_current]
        init_state_args['yaw_rate'] = self.ego_data['angular_rate'][t_current]
        init_state_args['slip_angle'] = self.ego_data['steering_angle'][t_current]
        return init_state_args
    
    def prepare_feature(self, iteration, ego_state=None):
        """根据当前轨迹数据准备模型推理所需的特征
        
        参数:
            iteration: 当前迭代次数，用于确定要提取特征的时间点
        
        返回:
            features: 包含模型所需特征的字典
            track_ids: 相关智能体的ID列表
        """
        # 定义参数
        num_agents = 20  # 最大智能体数量
        history_steps = 21  # 历史帧数量
        H = history_steps  # 历史帧简写
        
        # 当前时间点
        t_current = iteration + Offset
        self.iteration = t_current
           # 创建新组件
        if ego_state is not None:
            center_now = StateSE2(
                x=self.ego_data['x'][t_current],
                y=self.ego_data['y'][t_current],
                heading=self.ego_data['heading'][t_current]
            )
            velocity2d = StateVector2D(
                x=self.ego_data['vx'][t_current],
                y=self.ego_data['vy'][t_current]
            )
            acceleration2d = StateVector2D(
                x=self.ego_data['acceleration'][t_current],
                y=self.ego_data['ay'][t_current]
            )
            angular_velocity_now = self.ego_data['angular_rate'][t_current]
            tire_steering_angle_now = self.ego_data['steering_angle'][t_current]
            
            param = VehicleParameters(  
                                    width = W,
                                    front_length = L/2.0,
                                    rear_length = L/2.0,
                                    cog_position_from_rear_axle = 1.0,
                                    wheel_base = 2.65,
                                    vehicle_name = "ego",
                                    vehicle_type = "cs55")
            car_footprint = CarFootprint(
                center=center_now,
                vehicle_parameters=param,
            )
            
            # 创建新的DynamicCarState
            dynamic_car_state = DynamicCarState(
                rear_axle_to_center_dist = 2.65/2.0,
                rear_axle_velocity_2d = velocity2d,
                rear_axle_acceleration_2d = acceleration2d, 
                angular_velocity= angular_velocity_now,
                angular_acceleration = 0.0,
                tire_steering_rate = 0.0,
            )

            
            update_state = EgoState(
                car_footprint=car_footprint,
                dynamic_car_state=dynamic_car_state,
                is_in_auto_mode = True,
                tire_steering_angle=tire_steering_angle_now,
                time_point = ego_state.time_point,
            )
        # if ego_state is not None:
        #     ego_state.car_footprint.center.x = self.ego_data['x'][t_current]
        #     ego_state.car_footprint.center.y = self.ego_data['y'][t_current]
        #     ego_state.car_footprint.center.heading = self.ego_data['heading'][t_current]
        #     ego_state.dynamic_car_state.rear_axle_velocity_2d.x = self.ego_data['vx'][t_current]
        #     ego_state.dynamic_car_state.rear_axle_velocity_2d.y = self.ego_data['vy'][t_current]
        #     ego_state.dynamic_car_state.rear_axle_acceleration_2d.x = self.ego_data['ax'][t_current]
        #     ego_state.dynamic_car_state.rear_axle_acceleration_2d.y = self.ego_data['ay'][t_current]
        #     # ego_state.dynamic_car_state.angular_velocity = self.ego_data['angular_rate'][t_current]
        #     ego_state.car_footprint.width = W
        #     ego_state.car_footprint.length = L   
        #     ego_state.tire_steering_angle = self.ego_data['steering_angle'][t_current]     
        self._history_trajectory.append([self.ego_data['x'][t_current], self.ego_data['y'][t_current], self.ego_data['heading'][t_current]])
        # 提取ego当前状态
        ego_pose = np.array([self.ego_data['x'][t_current], self.ego_data['y'][t_current], self.ego_data['heading'][t_current]])
        # 构造变换矩阵
        cos_theta = np.cos(ego_pose[2])
        sin_theta = np.sin(ego_pose[2])
        translation_matrix = np.array([
            [1, 0, -ego_pose[0]],
            [0, 1, -ego_pose[1]],
            [0, 0, 1]
        ])
        rotation_matrix = np.array([
            [cos_theta, sin_theta, 0],
            [-sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        transform_matrix = np.dot(rotation_matrix, translation_matrix)
        
        # 提取ego历史轨迹
        ego_agent_past = np.zeros((H, 7), dtype=np.float32)
        for i in range(H):
            t_idx = t_current - H + i + 1
            if t_idx >= 0:
                ego_agent_past[i, 0] = self.ego_data['x'][t_idx]  # x
                ego_agent_past[i, 1] = self.ego_data['y'][t_idx]  # y
                ego_agent_past[i, 2] = self.ego_data['heading'][t_idx]  # heading
                ego_agent_past[i, 3] = self.ego_data['vx'][t_idx]  # vx
                ego_agent_past[i, 4] = self.ego_data['vy'][t_idx]  # vy
                ego_agent_past[i, 5] = self.ego_data['ax'][t_idx]  # ax
                ego_agent_past[i, 6] = self.ego_data['ay'][t_idx]  # ay
        # 根据ego_pose变换历史轨迹的x, y, heading, vx, vy, ax, ay
        # Translate coordinates relative to ego_pose
        # Transform historical coordinates to the current coordinate system
        for i in range(H):
            point = np.array([ego_agent_past[i, 0], ego_agent_past[i, 1], 1.0])  # Homogeneous coordinates
            transformed_point = np.dot(transform_matrix, point)
            ego_agent_past[i, 0] = transformed_point[0]
            ego_agent_past[i, 1] = transformed_point[1]
            # Transform heading
            ego_agent_past[i, 2] = angle_sub(ego_agent_past[i, 2], ego_pose[2])

            # Transform velocity
            ego_agent_past[i, 3] = ego_agent_past[i, 3] * cos_theta + ego_agent_past[i, 4] * sin_theta
            ego_agent_past[i, 4] = -ego_agent_past[i, 3] * sin_theta + ego_agent_past[i, 4] * cos_theta

            # Transform acceleration
            ego_agent_past[i, 5] = ego_agent_past[i, 5] * cos_theta + ego_agent_past[i, 6] * sin_theta
            ego_agent_past[i, 6] = -ego_agent_past[i, 5] * sin_theta + ego_agent_past[i, 6] * cos_theta
            
        # 提取npc历史轨迹 (假设只有一个npc)
        neighbor_agents_past = np.zeros((num_agents, H, 11), dtype=np.float32)  # [num_agents, history_steps, features]
        
        # 填充第一个智能体的特征 (只有一个npc)
        npc_agent = np.zeros((H, 11), dtype=np.float32)  # [history_steps, features]
        for i in range(H):
            t_idx = t_current - H + i + 1
            if t_idx >= 0:
                npc_agent[i, 0] = self.npc_data['x'][t_idx]
                npc_agent[i, 1] = self.npc_data['y'][t_idx]
                npc_agent[i, 2] = self.npc_data['heading'][t_idx]  # heading
                npc_agent[i, 3] = self.npc_data['vx'][t_idx]  # vx
                npc_agent[i, 4] = self.npc_data['vy'][t_idx]  # vy
                npc_agent[i, 5] = self.npc_data['yaw_rate'][t_idx]  # yaw rate
                npc_agent[i, 6] = L
                npc_agent[i, 7] = W
                npc_agent[i, 8:11] = np.array([1, 0, 0])  # 车辆类型编码 [vehicle, pedestrian, others]

        # 根据ego_pose变换npc历史轨迹的x, y, heading, vx, vy, yaw_rate
        # Transform historical coordinates to the current coordinate system
        for i in range(H):
            point = np.array([npc_agent[i, 0], npc_agent[i, 1], 1.0])
            transformed_point = np.dot(transform_matrix, point)
            npc_agent[i, 0] = transformed_point[0]
            npc_agent[i, 1] = transformed_point[1]
            # Transform heading
            npc_agent[i, 2] = angle_sub(npc_agent[i, 2], ego_pose[2])
            # Transform velocity
            npc_agent[i, 3] = npc_agent[i, 3] * cos_theta + npc_agent[i, 4] * sin_theta
            npc_agent[i, 4] = -npc_agent[i, 3] * sin_theta + npc_agent[i, 4] * cos_theta
        neighbor_agents_past[0] = npc_agent
        
        # 处理车道特征
        # 点的数量为200，下采样为50
        lanes_array = np.zeros((40, 50, 7), dtype=np.float32)  # [num_lanes, points_per_lane, features]
        for idx, lane_data in enumerate(self.lanes.values()):
            if idx >= 40:  # 最多处理40条车道
                break
                
            # 获取车道中心线
            center_points = lane_data["center"]
            
            # 转换到ego坐标系
            center_points_local = center_points.copy()
            points_to_use = min(50, len(center_points_local))
            point_index = np.linspace(0, len(center_points_local) - 1, points_to_use).astype(int)
            # print(point_index)
            center_points_local = center_points_local[point_index]
            # 坐标变换
            for i in range(points_to_use):
                point = np.array([center_points_local[i, 0], center_points_local[i, 1], 1.0])
                transformed_point = np.dot(transform_matrix, point)
                center_points_local[i, 0] = transformed_point[0]
                center_points_local[i, 1] = transformed_point[1]
                
            center_heading = np.zeros(points_to_use)
            for i in range(points_to_use):
                if i > 0:
                    center_heading[i] = np.arctan2(center_points_local[i, 1] - center_points_local[i-1, 1],
                                                    center_points_local[i, 0] - center_points_local[i-1, 0])
            center_heading[0] = center_heading[1]
            # print(center_heading)
            lanes_array[idx, :, :2] = center_points_local
            lanes_array[idx, :, 2] = center_heading  # 车道中心线的朝向
            lanes_array[idx, :, 3:7] = np.array([0, 0, 0, 1])  # 车道宽度和长度
        
        reference_path = None
        if iteration == 0:
            reference_path = np.array([self.route['x'][Offset:], self.route['y'][Offset:]]).T
        '''
        # 坐标变换
        for i in range(len(reference_path)):
            point = np.array([reference_path[i, 0], reference_path[i, 1], 1.0])
            transformed_point = np.dot(transform_matrix, point)
            reference_path[i, 0] = transformed_point[0]
            reference_path[i, 1] = transformed_point[1]
        plt.cla()
        plt.figure(1)
        time = np.arange(0, H*DT, DT)
        plt.plot(ego_agent_past[:, 0], ego_agent_past[:, 1], 'b')
        plt.plot(npc_agent[:, 0], npc_agent[:, 1], 'r')
        plt.plot(reference_path[:, 0], reference_path[:, 1], 'k')
        
        for idx, lane_data in enumerate(self.lanes.items()):
            plt.plot(lanes_array[idx, :10, 0], lanes_array[idx, :10, 1], 'g')
            break
        plt.axis('equal')
        plt.figure(2)
        plt.plot(time, ego_agent_past[:, 2], 'b')
        plt.plot(time, npc_agent[:, 2], 'r')
        plt.figure(3)
        plt.plot(time, ego_agent_past[:, 3], 'r')
        plt.plot(time, ego_agent_past[:, 4], 'b')
        plt.plot(time, npc_agent[:, 3], 'r')
        plt.plot(time, npc_agent[:, 4], 'b')
        plt.show()
        '''
            
        # route lanes
        route_lane_idx = 0
        route_lanss_array = np.zeros((10, 50, 3), dtype=np.float32)  # [num_lanes, points_per_lane, features]
        route_lanss_array[0, :, :] = lanes_array[route_lane_idx, :, :3]
        
        ego_agent_past = torch.tensor(ego_agent_past, dtype=torch.float32)
        neighbor_agents_past = torch.tensor(neighbor_agents_past, dtype=torch.float32)
        lanes_tensor = torch.tensor(lanes_array, dtype=torch.float32)  # [num_lanes, points_per_lane, features]
        route_lanss_tensor = torch.tensor(route_lanss_array, dtype=torch.float32)  # [num_lanes, points_per_lane, features]
        map_crosswalks_tensor = torch.zeros((5, 30, 3), dtype=torch.float32)  # [num_crosswalks, points_per_crosswalk, features]
        
        # 构建特征字典
        features = {
            "ego_agent_past": ego_agent_past.unsqueeze(0),  # 添加批次维度
            "neighbor_agents_past": neighbor_agents_past.unsqueeze(0),  # 添加批次维度
            "map_lanes": lanes_tensor.unsqueeze(0),  # 添加批次维度
            "map_crosswalks": map_crosswalks_tensor.unsqueeze(0),  # 添加批次维度
            "route_lanes": route_lanss_tensor.unsqueeze(0),  # 添加批次维度
        }
        
        # 返回特征和智能体ID
        track_ids = ["npc1"]  # 示例ID，实际情况可能有多个
        target_speed = 5.0
        current_speed = self.ego_data['velocity'][t_current]
        target_speed = max(target_speed, current_speed)
        # 如果需要转换到GPU
        if self._device != 'cpu':
            features = {k: v.to(self._device) for k, v in features.items()}
            
        return features, track_ids, reference_path, target_speed, update_state
    def plot_scenario(self, predictions=None, planning_trajectory=None, candidate_trajectories=None, return_img=True):
        """绘制完整场景，包括车辆、车道和轨迹"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ego_pose = self._history_trajectory[-1]
        cos_theta = np.cos(ego_pose[2])
        sin_theta = np.sin(ego_pose[2])
        translation_matrix = np.array([
            [1, 0, -ego_pose[0]],
            [0, 1, -ego_pose[1]],
            [0, 0, 1]
        ])
        rotation_matrix = np.array([
            [cos_theta, sin_theta, 0],
            [-sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        transform_matrix = np.dot(rotation_matrix, translation_matrix)

        # 绘制车道
        kwargs = {"color": "lightgray", "alpha": 0.2, "ec": None, "zorder": 0}
        for lane in self.lanes.values():
            point_index = np.linspace(0, len(lane["left_boundary"]) - 1, 100).astype(int)
            left_boundary = np.array(lane["left_boundary"][point_index])
            right_boundary = np.array(lane["right_boundary"][point_index])
            center_line = np.array(lane["center"][point_index])
            # 转换到ego坐标系
            for i in range(len(left_boundary)):
                point = np.array([left_boundary[i, 0], left_boundary[i, 1], 1.0])
                transformed_point = np.dot(transform_matrix, point)
                left_boundary[i, 0] = transformed_point[0]
                left_boundary[i, 1] = transformed_point[1]
            for i in range(len(right_boundary)):
                point = np.array([right_boundary[i, 0], right_boundary[i, 1], 1.0])
                transformed_point = np.dot(transform_matrix, point)
                right_boundary[i, 0] = transformed_point[0]
                right_boundary[i, 1] = transformed_point[1]
            for i in range(len(center_line)):
                point = np.array([center_line[i, 0], center_line[i, 1], 1.0])
                transformed_point = np.dot(transform_matrix, point)
                center_line[i, 0] = transformed_point[0]
                center_line[i, 1] = transformed_point[1]
            
            # 创建并添加车道多边形
            lane_polygon = np.vstack((left_boundary[:], right_boundary[:][::-1]))
            ax.add_artist(self._polygon_to_patch(lane_polygon, **kwargs))
            cl_color, linewidth = "gray", 1.0
            ax.plot(
                center_line[:, 0],
                center_line[:, 1],
                color=cl_color,
                alpha=0.5,
                linestyle="--",
                zorder=1,
                linewidth=linewidth,
            )
        
        # 绘制ego车辆
        kwargs = {"lw": 1.5, "ec": "red", "fill": False, "zorder": 10, "alpha": 1.0}
        ego_polygon = self._get_vehicle_polygon(np.array([0,0]), 0, L, W)
        ax.add_artist(self._polygon_to_patch(ego_polygon, **kwargs))
        ax.plot(
            [0, 0 + L /1.5],
            [0, 0],
            color="red",
            linewidth=1.5,
            zorder=11,
        )
        if planning_trajectory is not None:
            self._plot_planning(ax, planning_trajectory)
        if candidate_trajectories is not None:
            self._plot_candidate_trajectories(ax, candidate_trajectories)
        
        # plot history trajectory
        history_trajectory = np.zeros((len(self._history_trajectory), 2), dtype=np.float32) 
        for i in range(len(self._history_trajectory)):
            # Transform the coordinates
            point = np.array([self._history_trajectory[i][0], self._history_trajectory[i][1], 1.0])
            transformed_point = np.dot(transform_matrix, point)
            history_trajectory[i, 0] = transformed_point[0]
            history_trajectory[i, 1] = transformed_point[1]
        ax.plot(
            history_trajectory[:, 0],
            history_trajectory[:, 1],
            color="red",
            alpha=0.5,
            zorder=6,
            linewidth=2,
        )
        
        t_current = self.iteration
        # 绘制NPC车辆
        kwargs = {"lw": 1.5, "ec": "#001eff", "fill": False, "zorder": 4, "alpha": 1.0}
        npc_pos = np.array([self.npc_data['x'][t_current], self.npc_data['y'][t_current]])
        npc_heading = self.npc_data['heading'][t_current]
        # 坐标变换
        point = np.array([npc_pos[0], npc_pos[1], 1.0])
        transformed_point = np.dot(transform_matrix, point)
        npc_pos[0] = transformed_point[0]
        npc_pos[1] = transformed_point[1]
        # Transform heading
        npc_heading = angle_sub(npc_heading, ego_pose[2])
    
        npc_polygon = self._get_vehicle_polygon(npc_pos, npc_heading, L, W)
        ax.add_artist(self._polygon_to_patch(npc_polygon, **kwargs))
        direction_vector = np.array([npc_pos[0] + L / 1.5 * np.cos(npc_heading), npc_pos[1] + L / 1.5 * np.sin(npc_heading)])
        ax.plot(
            [npc_pos[0], direction_vector[0]],
            [npc_pos[1], direction_vector[1]],
             color="#001eff", 
             linewidth=1, 
             zorder=4
        )
        
        if predictions is not None:
            self._plot_prediction(ax, predictions)

        ax.axis("equal")
        bounds=60
        offset=20
        ax.set_xlim(xmin=-bounds + offset, xmax=bounds + offset)
        ax.set_ylim(ymin=-bounds, ymax=bounds)
        ax.invert_yaxis()  # 添加这一行
        ax.axis("off")
        plt.tight_layout(pad=0)
        if return_img:
            fig.canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
                int(height), int(width), 3
            )
            plt.close(fig)
            return img
        else:
            plt.show()
        
    def _polygon_to_patch(self, polygon_points, **kwargs):
        """将多边形点集转换为matplotlib的Patch对象
        
        参数:
            polygon_points: 多边形顶点坐标数组，形状为(N, 2)
            **kwargs: 传递给Polygon的额外参数，如颜色、透明度等
        
        返回:
            matplotlib.patches.Polygon对象
        """
        import matplotlib.patches as mpatches
        
        # 设置默认样式
        default_style = {
        }
        
        # 使用用户提供的参数覆盖默认样式
        for key, value in kwargs.items():
            default_style[key] = value
        
        # 创建多边形补丁
        polygon_patch = mpatches.Polygon(polygon_points, **default_style)
        
        return polygon_patch

    def _get_vehicle_polygon(self, position, heading, length, width):
        """生成车辆多边形"""
        corners = np.array([
            [length/2, width/2],
            [length/2, -width/2],
            [-length/2, -width/2],
            [-length/2, width/2]
        ])
        
        # 旋转
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rotation_matrix = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])
        
        rotated_corners = np.dot(corners, rotation_matrix.T)
        
        # 平移
        vehicle_polygon = rotated_corners + position
        
        return vehicle_polygon
        

if __name__ == '__main__':
    ego_csv = '/root/xzcllwx_ws/GameFormer-Planner/Exp3/cs55_record.csv'
    npc_csv = '/root/xzcllwx_ws/GameFormer-Planner/Exp3/npc_record.csv'
    location = "cuda:0"
    planner = SceneLoader(ego_csv_file=ego_csv, npc_csv_file=npc_csv ,device=location)
    planner.plot_data()
    # 保存self.ego_data["acceleration"]的值为csv
    ego_acceleration = planner.ego_data["acceleration"][Offset:Offset+145]
    np.savetxt('/root/xzcllwx_ws/GameFormer-Planner/Exp3/ego_acceleration.csv', ego_acceleration, delimiter=',')

    # The parameters of the model should be the same as the one used in training
    model = GameFormer(encoder_layers=3, decoder_levels=2)
        
    # Load trained model
    model.load_state_dict(torch.load("/root/xzcllwx_ws/GameFormer-Planner/training_log/Exp3/model_epoch_30_valADE_1.1357.pth", 
                                map_location=location))
    features, track_ids, reference_path = planner.prepare_feature(iteration=0)
    model.to(location)
    model.eval()
    with torch.no_grad():
        predictions, plan = model(features)
        K = len(predictions) // 2 - 1
        final_predictions = predictions[f'level_{K}_interactions'][:, 1:]
        final_scores = predictions[f'level_{K}_scores']
        scores = torch.nn.functional.softmax(final_scores, dim=-1)
    
    with torch.no_grad():  
        predictions = final_predictions
        _, N, _, _, _ = predictions.shape
        scores = scores.squeeze(0)  # 移除批次维度，变成 [N, M]
        scores =  scores[1:, :]  # 移除第一个agent的预测 → 对于预测模型，需要确认
        predictions = predictions.squeeze(0)  # 移除批次维度，变成 [N, M, T, D]
        
        best_indices = torch.argmax(scores, dim=1)  # 维度 [N]
        agent_indices = torch.arange(N)  # 维度 [N]

        best_predictions = predictions[agent_indices, best_indices]  # 维度 [N, T, D]
        agent_current = features['neighbor_agents_past'][0][:N, -1, :]
        current_positions = agent_current[:, :2].unsqueeze(1)  # Extract current positions and add time dimension
        current_sigma_x = torch.full((N, 1), 0.001, device=best_predictions.device)  # sigma_x
        current_sigma_y = torch.full((N, 1), 0.001, device=best_predictions.device)  # sigma_y
        current_rho = torch.full((N, 1), 0.0, device=best_predictions.device)  # rho
        current_covariances = torch.cat([current_sigma_x, current_sigma_y, current_rho], dim=1).unsqueeze(1)  # Add time dimension

        # Concatenate current positions and covariances to the beginning of best_predictions
        best_predictions = torch.cat([torch.cat([current_positions, current_covariances], dim=2), best_predictions], dim=1)
        
        output_predictions = best_predictions.cpu().numpy()
    planner.plot_scenario(predictions=output_predictions)

