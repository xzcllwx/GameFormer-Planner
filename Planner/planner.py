import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
import math
import time
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.geometry.base import CAP_STYLE
from .planner_utils import *
from .smoother import MotionNonlinearSmoother
from .observation import *
from GameFormer.predictor import GameFormer
from .state_lattice_path_planner import LatticePlanner

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


import copy
sys.path.append(os.path.abspath('/root/xzcllwx_ws/GameFormer-Planner/iLQR/build/'))
import motion_planning
sys.path.append("/root/xzcllwx_ws")
from pluto.src.utils.vis import *
from pluto.src.feature_builders.nuplan_scenario_render import *
from pluto.src.scenario_manager.scenario_manager import ScenarioManager
from EthicalTrajectoryPlanning.planner.Frenet.frenet_planner import FrenetPlanner
from EthicalTrajectoryPlanning.planner.Frenet.configs.load_json import (
    load_harm_parameter_json,
    load_planning_json,
    load_risk_json,
    load_weight_json,
)
from EthicalTrajectoryPlanning.planner.utils.vehicleparams import VehicleParameters
from EthicalTrajectoryPlanning.planner.utils.timers import ExecTimer


from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.geometry.shape import Rectangle, Circle, Polygon, ShapeGroup, Shape
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.common.util import Interval
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D




class Planner(AbstractPlanner):
    def __init__(self, model_path, device=None):
        self._max_path_length = MAX_LEN # [m]
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._target_speed = 13.0 # [m/s]
        self._N_points = int(T/DT)
        self._model_path = model_path

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self._device = device
        
        self._render = False
        self._time = False
        self._N_agent = 5
        self._N_static_obstacle = 0
        # load settings from planning_fast.json
        settings_dict = load_planning_json("planning_fast.json")
        settings_dict["risk_dict"] = risk_dict = load_risk_json()
        if not self._time:
            settings_dict["evaluation_settings"]["show_visualization"] = True 
        self.settings = settings_dict
        self.vehicle_params = VehicleParameters(settings_dict["evaluation_settings"]["vehicle_type"])
        self.exec_timer = ExecTimer(timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"])
        self.frenet_settings = settings_dict["frenet_settings"]
        self.config_path = '/root/xzcllwx_ws/GameFormer-Planner/iLQR/config/scenario_two_borrow.yaml'
        self.csv_save_dir = '/root/xzcllwx_ws/GameFormer-Planner/csv'

        self.s_positive_threhold = 5
        self.s_negative_threhold = -10
        self.d_threhold = 5
        
        
    def name(self) -> str:
        return "GameFormer Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._initialization = initialization
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal # mission goal StateSE2
        self._route_roadblock_ids = initialization.route_roadblock_ids
        # self._initialize_route_plan(self._route_roadblock_ids)
        # self._initialize_model()
        self._confingency_planner = None
        self._trajectory_planner = TrajectoryPlanner()
        self._path_planner = None
        self._scenario_manager = None
        self._scene_render = NuplanScenarioRender()
        self._imgs = []
        self._save_dir = '/root/xzcllwx_ws/GameFormer-Planner/figure'
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        self._motion_planner = motion_planning.motion_planner(self.config_path)
        
        self._global_plan = None
        
        self._reference_spline = None
        self._target_speed = 13.0
        
        self._last_track_target = None
        self._last_belief = {"leader": [], "follower": [], "begin_iteration": 0, "history_x": [], "history_y": [], "branch_time": []}
        self._last_pred = {"leader": None, "follower": None}
        self._smoother = MotionNonlinearSmoother(trajectory_len=T*10-1, dt=DT)
        self.frame_data = {}
        self.save_data = True
        self._ego_state = {
            "x": [],
            "y": [],
            "heading": [],
            "velocity": [],
            "acceleration": [],
            "yaw_rate": [],
            "steer_angle": [],
            "steering_rate": [],            
        }


    def _initialize_model(self):
        # The parameters of the model should be the same as the one used in training
        self._model = GameFormer(encoder_layers=3, decoder_levels=2)
        
        # Load trained model
        self._model.load_state_dict(torch.load(self._model_path, map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        
    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]
    
    
    def _add_obstacle(self):
        init_state_args = dict()
        init_state_args['position'] = np.array([0, 0])
        init_state_args['orientation'] = 0
        init_state_args['velocity'] = 0
        init_state_args['acceleration'] = 0
        init_state_args['yaw_rate'] = 0
        init_state_args['slip_angle'] = 0
        init_state_args['time_step'] = 0
        init_state = State(**init_state_args)
        obstacle_shape = Rectangle(width=self._car_w, length=self._car_l)
        ego_obstacle = DynamicObstacle(
            obstacle_id=0,
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=obstacle_shape,
            initial_state=init_state,
        )
        self._scenario.add_objects(ego_obstacle)  
        
        for i in range(1, self._N_agent+1):
            agent_obstacle = DynamicObstacle(
                obstacle_id=i,
                obstacle_type=ObstacleType.CAR,
                obstacle_shape=obstacle_shape,
                initial_state=init_state,
            )
            self._scenario.add_objects(agent_obstacle)
            
            
        for i in range(self._N_agent+1, self._N_agent + self._N_static_obstacle):
            obstacle_shape = Rectangle(width=2, length=2)
            obstacle = DynamicObstacle(
                obstacle_id=i,
                obstacle_type=ObstacleType.PARKED_VEHICLE,
                obstacle_shape=obstacle_shape,
                initial_state=init_state,
            )
            self._scenario.add_objects(obstacle)
    
    def _init_laneletnets(self, initial_state):
        self._scenario = Scenario(
            dt = DT,
            scenario_id = '0',
        )
       
        position_list = list()
        lanelets_of_goal_position = dict()
        for id in self._candidate_lane_edge_ids:
            # print(f'lane id: {id}')
            lane = self._map_api.get_map_object(id, SemanticMapLayer.LANE)
            lane = lane or self._map_api.get_map_object(id, SemanticMapLayer.LANE_CONNECTOR)

            # bound [[x0,x1,...,xn],[y0,y1,...,yn]]
            # 创建指定形状的空数组
            point_count = len(lane.baseline_path.discrete_path)
            # print(f'base point count: {point_count}')
            base_line = np.ndarray(shape=(2, point_count), dtype=np.float32)
            base_line[0] = [p.x for p in lane.baseline_path.discrete_path]
            base_line[1] = [p.y for p in lane.baseline_path.discrete_path]
            base_line = base_line.T 
            
            point_count = len(lane.left_boundary.discrete_path)
            # print(f'left point count: {point_count}')
            left_bound = np.ndarray(shape=(2, point_count), dtype=np.float32)
            left_bound[0] = [p.x for p in lane.left_boundary.discrete_path]
            left_bound[1] = [p.y for p in lane.left_boundary.discrete_path]
            left_bound = left_bound.T
            
            point_count = len(lane.right_boundary.discrete_path)
            # print(f'right point count: {point_count}')
            right_bound = np.ndarray(shape=(2, point_count), dtype=np.float32)
            right_bound[0] = [p.x for p in lane.right_boundary.discrete_path]
            right_bound[1] = [p.y for p in lane.right_boundary.discrete_path]
            right_bound = right_bound.T
             
            lanelet = Lanelet(
                left_vertices = left_bound, 
                center_vertices = base_line, 
                right_vertices = right_bound, 
                lanelet_id = int(lane.id))
            self._scenario.lanelet_network.add_lanelet(lanelet)
            polygon = lanelet.convert_to_polygon()
            position_list.append(polygon)
            if 0 not in lanelets_of_goal_position:
                lanelets_of_goal_position[0] = []
            lanelets_of_goal_position[0].append(lanelet.lanelet_id)

        # creat planning problem
        init_state_args = dict()
        init_state_args['position'] = np.array([initial_state.car_footprint.center.x, initial_state.car_footprint.center.y])
        init_state_args['orientation'] = initial_state.car_footprint.center.heading
        init_state_args['velocity'] = initial_state.dynamic_car_state.rear_axle_velocity_2d.x
        init_state_args['acceleration'] = initial_state.dynamic_car_state.rear_axle_acceleration_2d.x
        init_state_args['yaw_rate'] = initial_state.dynamic_car_state.angular_velocity
        init_state_args['slip_angle'] = initial_state.dynamic_car_state.tire_steering_rate
        init_state_args['time_step'] = Interval(0, 0) # ms
        init_state = State(**init_state_args)
        position = ShapeGroup(position_list)
        state_args = dict()
        state_args['position'] = position
        state_args['time_step'] = Interval(0, 100) # ms
        goal_state = State(**state_args)
        state_list = []
        state_list.append(goal_state)
        goal_region = GoalRegion(state_list, lanelets_of_goal_position)
        self._planning_problem = PlanningProblem(
            planning_problem_id = 0,
            initial_state = init_state,
            goal_region = goal_region,
        )
        
        self._car_w = initial_state.car_footprint.width
        self._car_l = initial_state.car_footprint.length
        # add obstacle to scenario
        self._add_obstacle()
        
        # create frenet planner
        if self._confingency_planner is None:
            self._confingency_planner = FrenetPlanner(
                scenario=self._scenario,
                planning_problem=self._planning_problem,
                ego_id=0,
                vehicle_params=self.vehicle_params,
                mode=self.frenet_settings["mode"],
                frenet_parameters=self.frenet_settings["frenet_parameters"],
                settings=self.settings,
                plot_frenet_trajectories=True,
            )

    def _get_reference_path(self, ego_state, traffic_light_data, observation):
        # Get starting block
        starting_block = None
        min_target_speed = 3
        max_target_speed = 15
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
            
        # In case the ego vehicle is not on the route, return None
        if closest_distance > 5:
            return None

        # Get reference path, handle exception
        try:
            ref_path, optimal_origin_path = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        except:
            ref_path = None

        if ref_path is None:
            return None

        # Annotate red light to occupancy
        # occupancy = np.zeros(shape=(ref_path.shape[0], 1))
        # for data in traffic_light_data:
        #     id_ = str(data.lane_connector_id)
        #     if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
        #         lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
        #         conn_path = lane_conn.baseline_path.discrete_path
        #         conn_path = np.array([[p.x, p.y] for p in conn_path])
        #         red_light_lane = transform_to_ego_frame(conn_path, ego_state)
        #         occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

        # Annotate max speed along the reference path
        target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        # max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        # ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]
        # if len(ref_path) < MAX_LEN * 10:
        #     ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10-len(ref_path), axis=0), axis=0)
        ref_path = ref_path[:, :2]
        # plt.plot(ref_path[:, 0].copy(), ref_path[:, 1].copy(), 'r')
        # plt.show()
        if len(ref_path) < MAX_LEN * 10:
            extend_len = int(MAX_LEN * 10 - len(ref_path))
            extend_point_num = int(extend_len / 0.1 )
            last_point = ref_path[-1]
            if len(ref_path) >= 5:
                direction = ref_path[-1, :2] - ref_path[-5, :2]  # 计算方向向量
                # 沿着该方向添加新点
                direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
                extended_points = np.array([last_point + direction * (i+1) * 0.1
                                        for i in range(extend_point_num)])
            else:
                # 只有一个点时无法确定方向，只能重复
                extended_points = np.repeat(last_point[np.newaxis, :], MAX_LEN*10-len(ref_path), axis=0)
            ref_path = np.append(ref_path, extended_points, axis=0)
        
        extend_len = REVERSE_EXTEND_LEN
        extend_point_num = int(extend_len / 0.1) 
        first = ref_path[0]
        if len(ref_path) >= 1:
            direction = ref_path[0, :2] - ref_path[1, :2]  # Use the direction between the first two points
            direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
            extended_points = np.array([first + direction * (i + 1) * 0.1
                        for i in range(extend_point_num)])
        else:
            extended_points = np.repeat(first[np.newaxis, :], extend_point_num, axis=0)
        extended_points = extended_points[::-1]
        ref_path = np.append(extended_points, ref_path, axis=0)
        
        # plt.plot(ref_path[:, 0], ref_path[:, 1], 'b')
        # plt.axis('equal')
        # plt.show()
        return ref_path.astype(np.float32), target_speed, optimal_origin_path

    def _get_prediction(self, features):
        predictions, plan = self._model(features)
        K = len(predictions) // 2 - 1
        final_predictions = predictions[f'level_{K}_interactions'][:, 1:]
        final_scores = predictions[f'level_{K}_scores']
        ego_current = features['ego_agent_past'][:, -1]
        neighbors_current = features['neighbor_agents_past'][:, :, -1]
        final_scores = torch.nn.functional.softmax(final_scores, dim=-1)

        return plan, final_predictions, final_scores, ego_current, neighbors_current
    
    def _get_constant_speed_prediction(self, features):
        agents = features['neighbor_agents_past'][0]
        N = len(agents)
        T = self._N_points
        device = agents.device
        feature_dim = 5  # 获取特征维度
        
        last_states = agents[:, -1, :]
        # The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate) and size (length, width) at time t.
        pos = last_states[:, :2]
        heading = last_states[:, 2]
        speed = torch.norm(last_states[:, 3:5], dim=1, keepdim=True)  # [N, 1]

        dt = torch.tensor(DT, device=device)
        time_step = torch.arange(1, T + 1, device=device).float() * dt
        time_steps = time_step.view(1, T, 1).expand(N, T, 1)  # [N, T, 1]

        cos_h = torch.cos(heading).unsqueeze(1)  # [N, 1]
        sin_h = torch.sin(heading).unsqueeze(1)  # [N, 1]

        speed_x = speed * cos_h 
        speed_y = speed * sin_h
        speed_x_expand = speed_x.unsqueeze(1).expand(N, T, 1)  # [N, T, 1]
        speed_y_expand = speed_y.unsqueeze(1).expand(N, T, 1)  # [N, T, 1]
        future_x = pos[:, 0].unsqueeze(1) + (speed_x_expand * time_steps).squeeze(-1)  # [N, T, 1]
        future_y = pos[:, 1].unsqueeze(1) + (speed_y_expand * time_steps).squeeze(-1)  # [N, T, 1]
        future_x.unsqueeze_(-1)
        future_y.unsqueeze_(-1)
        future_pos = torch.cat([future_x, future_y], dim=2)  # [N, T, 2]

        predictions = torch.zeros(1, N, T, feature_dim, device=device)  # [1, N, T, D]

        predictions[0, :, :, :2] = future_pos
        predictions[0, :, :, 2:4] = 1  # cov
        predictions[0, :, :, 4] = 0.1  # rho

        predictions.unsqueeze_(2)  # Add a modality dimension, shape becomes [1, N, T, D]
        valid_mask = ~torch.eq(last_states.sum(-1), 0).unsqueeze(1).unsqueeze(2).unsqueeze(2)  # [N, 1, 1]
        predictions[0] = predictions[0] * valid_mask  # Automatically broadcast to all time steps
        
        scores = torch.ones(1, N, 1, 1, device=device)
        scores[0] = scores[0] * valid_mask.squeeze(-1)
        scores.squeeze_(-1)
        
        return predictions, scores  
    
    def _update_scenario_by_feature(self, features, N_agent):
        for i in range(N_agent):
            agent = features['neighbor_agents_past'][0][i]
            if torch.eq(agent[-1].sum(-1), 0):
                return i
            length = agent[-1][6].cpu().item()
            width = agent[-1][7].cpu().item()
            object = self._scenario.obstacle_by_id(i+1)
            obstacle_shape = Rectangle(width=width, length=length)
            if object is not None:
                object.obstacle_shape = obstacle_shape
        return N_agent
    
    def _plan(self, iteration, ego_state, history, traffic_light_data, observation):
        
        rotation = ego_state.car_footprint.center.heading
        translation = np.array([ego_state.car_footprint.center.x, ego_state.car_footprint.center.y]).reshape(1, 2)
        rot_mat = np.array(
            [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
        )
        if self.save_data:
            self.frame_data = {}
            self.frame_data['iteration'] = iteration
            self.frame_data['timestamp'] = iteration * DT
        
        # Construct input features
        # 转换到了ego坐标系下
        features, track_ids = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device) 

        # update the scenario by feature
        valid_agent_num = self._update_scenario_by_feature(features, self._N_agent)

        # Get reference path
    
        if self._reference_spline is None:
            print("init reference path")
            ref_path, target_speed, optimal_origin_path = self._get_reference_path(ego_state, traffic_light_data, observation)
            self._target_speed = target_speed
            # Transform reference path
            global_path = ref_path[:,:2]
            global_path = np.matmul(global_path, rot_mat.T)
            global_path = global_path + translation
            # Downsample global_path to 1m intervals
            downsampled_indices = np.arange(0, len(global_path), 10)
            global_path = global_path[downsampled_indices]
            self._reference_spline = CubicSpline2D(x=global_path[:, 0], y=global_path[:, 1])
            self._confingency_planner.init_global_path(global_path)
            if self.save_data:
                self.frame_data['global_path_x'] = global_path[:, 0].tolist()
                self.frame_data['global_path_y'] = global_path[:, 1].tolist()
            # index = np.arange(0, len(optimal_origin_path), 4)
            # optimal_origin_path = optimal_origin_path[index]
            # global_path_2 = np.transpose(optimal_origin_path, (1, 0))
            global_path_2 = np.transpose(global_path, (1, 0))
            if self._motion_planner.set_reference_line(global_path_2.astype(np.float64)):
                print("Set reference line successfully!")
            else:
                print("Set reference line failed!")

        target_speed = self._target_speed
        if self.save_data:
            self.frame_data['target_speed'] = target_speed
        # Infer prediction model
        with torch.no_grad():
            # plan, predictions, scores, ego_state_transformed, neighbors_state_transformed = self._get_prediction(features)
            predictions, scores = self._get_constant_speed_prediction(features)
            
        with torch.no_grad():  
            _, N, _, _, _ = predictions.shape
            scores = scores.squeeze(0)  # 移除批次维度，变成 [N, M]
            # scores =  scores[1:, :]  # 移除第一个agent的预测 → 对于预测模型，需要确认
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
            
            # Transform positions
            trans_tensor = torch.tensor(translation, device=predictions.device, dtype=torch.float32).reshape(1, 2)
            rot_mat_tensor = torch.tensor(rot_mat, device=predictions.device, dtype=torch.float32).reshape(2, 2)
            best_predictions[:, :, :2] = torch.matmul(best_predictions[:, :, :2], rot_mat_tensor.T)
            best_predictions[:, :, :2] = best_predictions[:, :, :2] + trans_tensor
            
            # Extract sigma_x, sigma_y, and rho
            sigma_x = best_predictions[:, :,2]
            sigma_y = best_predictions[:, :,3]
            rho = best_predictions[:,:, 4]

            # Compute covariance matrices
            cov_matrices = torch.zeros((best_predictions.shape[0], best_predictions.shape[1],  2,  2), device=predictions.device, dtype=torch.float32)
            cov_matrices[:, :, 0, 0] = sigma_x ** 2
            cov_matrices[:, :, 1, 1] = sigma_y ** 2
            cov_matrices[:, : ,0, 1] = rho * sigma_x * sigma_y #需要对换位置，由于数值一样，无所谓
            cov_matrices[:, : ,1, 0] = rho * sigma_x * sigma_y

            # Apply rotation to covariance matrices
            cov_matrices = torch.matmul(rot_mat_tensor, torch.matmul(cov_matrices, rot_mat_tensor.T))
            
            state_args = dict()
            state_args['position'] = np.array([ego_state.car_footprint.center.x, ego_state.car_footprint.center.y])
            state_args['orientation'] = ego_state.car_footprint.center.heading
            state_args['velocity'] = np.sqrt(ego_state.dynamic_car_state.rear_axle_velocity_2d.x**2 + ego_state.dynamic_car_state.rear_axle_velocity_2d.y**2)
            state_args['acceleration'] = np.sqrt(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x**2 + ego_state.dynamic_car_state.rear_axle_acceleration_2d.y**2)
            state_args['yaw_rate'] = ego_state.dynamic_car_state.angular_velocity
            state_args['slip_angle'] = None
            state_args['time_step'] = iteration
            current_state = State(**state_args)
            if self.save_data:
                self._ego_state['x'].append(ego_state.car_footprint.center.x)
                self.frame_data['ego_x'] = self._ego_state['x']
                self._ego_state['y'].append(ego_state.car_footprint.center.y)
                self.frame_data['ego_y'] = self._ego_state['y']
                self._ego_state['heading'].append(ego_state.car_footprint.center.heading)
                self.frame_data['ego_heading'] = self._ego_state['heading']
                self._ego_state['velocity'].append(np.sqrt(ego_state.dynamic_car_state.rear_axle_velocity_2d.x**2 + ego_state.dynamic_car_state.rear_axle_velocity_2d.y**2))
                self.frame_data['ego_velocity'] = self._ego_state['velocity']
                self._ego_state['acceleration'].append(np.sqrt(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x**2 + ego_state.dynamic_car_state.rear_axle_acceleration_2d.y**2))
                self.frame_data['ego_acceleration'] = self._ego_state['acceleration']
                self._ego_state['yaw_rate'].append(ego_state.dynamic_car_state.angular_velocity)
                self.frame_data['ego_yaw_rate'] = self._ego_state['yaw_rate']
                self._ego_state['steer_angle'].append(ego_state.tire_steering_angle)
                self.frame_data['ego_steer_angle'] = self._ego_state['steer_angle']
                self.frame_data['ego_steering_rate'] = ego_state.dynamic_car_state.tire_steering_rate
                self.frame_data['ego_width'] = ego_state.car_footprint.width
                self.frame_data['ego_length'] = ego_state.car_footprint.length
            
            global_predictions = best_predictions[:, :, :2].cpu().numpy()
            # Calculate yaw for each agent's trajectory
            yaw = np.zeros_like(global_predictions[:, :, 0])
            yaw[:, :-1] = np.arctan2(
                np.diff(global_predictions[:, :, 1], axis=1),
                np.diff(global_predictions[:, :, 0], axis=1)
            )
            yaw[:, -1] = yaw[:, -2]
            yaw = yaw[:, :, np.newaxis]  # Add a new axis for yaw
            global_predictions = np.concatenate([global_predictions, yaw], axis=2)  # [N, T, D], where D includes x, y, yaw
            global_predictions = np.transpose(global_predictions, (0, 2, 1))  # [N, D, T]
            cov = cov_matrices[:].cpu().numpy()
            ego_s, _ = self._reference_spline.get_min_arc_length([current_state.position[0], current_state.position[1]])
            ego_yaw = current_state.orientation
            target_index = []
            d_square_threshold = (1.5*3.4)**2
            for i, pred in enumerate(global_predictions):
                if i >= len(track_ids):
                    break
                position = np.array([pred[0, 0], pred[1, 0]])
                agent_v = np.array([pred[0, 1] - pred[0, 0], pred[1, 1] - pred[1, 0]]) / DT
                agent_v_norm = np.linalg.norm(agent_v)
                agent_yaw = pred[2, 0]
                diff_yaw = angle_sub(ego_yaw, agent_yaw)
                if agent_v_norm > 5 and np.abs(diff_yaw) > np.pi / 2:
                    continue
                current_s, current_d_square = self._reference_spline.get_min_arc_length([position[0], position[1]])
                if current_s-ego_s < -50 or \
                    current_s-ego_s > 60 or \
                    current_d_square > d_square_threshold:
                    continue
                target_index.append(i)

            print(f'target index num: {len(target_index)}')

            if len(target_index) == 0:
                target_index = [0]
            if len(target_index) <= valid_agent_num:
                global_predictions = global_predictions[target_index]
                cov = cov[target_index]
                track_ids = [track_ids[i] for i in target_index]
            else:
                global_predictions = global_predictions[target_index[:valid_agent_num]]
                cov = cov[target_index[:valid_agent_num]]
                track_ids = [track_ids[i] for i in target_index[:valid_agent_num]]
                
            print(f'pred agent num: {global_predictions.shape[0]}')
            print(f'track agent num: {len(track_ids)}')
            
            if global_predictions.shape[0] == 0:
                global_predictions = np.zeros((1, 3, self._N_points + 1))
                cov = np.zeros((1, self._N_points + 1, 2, 2))
                for i in range(self._N_points + 1):
                    cov[0, i, 0, 0] = 0.01
                    cov[0, i, 1, 1] = 0.01

            track_dict = dict()
            for i, track_id in enumerate(track_ids):
                track_dict[track_id] = i+1

            global_predictions = global_predictions[:, np.newaxis, :, :]  # Add a modality axis after N
            cov = cov[:, np.newaxis, :, :, :]  # Add a modality axis after N

            # 博弈预测
            belief = [1]
            l_pred_x = None
            l_pred_y = None
            f_pred_x = None
            f_pred_y = None
            if self._global_plan is not None:
                line1 = LineString(self._global_plan[:, :2]).buffer((WIDTH/2), cap_style=CAP_STYLE.square)
                for i in range(len(global_predictions)):
                    agent_prediction = global_predictions[i][0] # [D, T]
                    position = np.array([agent_prediction[0, 0], agent_prediction[1, 0]])
                    v = np.array([agent_prediction[0, 1] - agent_prediction[0, 0], agent_prediction[1, 1] - agent_prediction[1, 0]]) / DT
                    v_norm = np.linalg.norm(v)
                    if v_norm < 1:
                        print(f'Agent {i+1} is too slow')
                        continue
                    current_s, current_d_square = self._reference_spline.get_min_arc_length([position[0], position[1]])
                    if current_s-ego_s < self.s_negative_threhold or \
                        current_s-ego_s > self.s_positive_threhold or \
                        math.sqrt(current_d_square) > self.d_threhold:
                        print(f'Agent {i+1} is out of range')
                        print(f'current_s-ego_s: {current_s-ego_s}')
                        print(f'current_d_square: {current_d_square}')
                        continue
                    line2 = LineString(agent_prediction[:2].T).buffer((WIDTH/2), cap_style=CAP_STYLE.square)
                    is_intersect = line1.intersects(line2)
                    if is_intersect:
                        # global_predictions模态轴复制
                        global_predictions = global_predictions.repeat(2, axis=1)
                        cov = cov.repeat(2, axis=1)
                        agent_prediction = global_predictions[i][0] # [D, T]
                        intersection_polygon = line1.intersection(line2)
                        agent_collsion_t, _ = find_intersection_indices(agent_prediction, intersection_polygon)
                        ego_collsion_t, _ = find_intersection_indices(self._global_plan, intersection_polygon)
                        ego_v = np.array([current_state.velocity*np.cos(current_state.orientation), current_state.velocity*np.sin(current_state.orientation)])
                        
                        intersection_points = np.array([agent_prediction[0, agent_collsion_t], agent_prediction[1, agent_collsion_t]])
                        tangent_vector = np.array([intersection_points[0]-agent_prediction[0, 0], intersection_points[1]-agent_prediction[1, 0]])
                        tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
                        print(f'Agent {i+1} is intersecting with global plan and track token is {track_ids[i]}')
                        if self._last_track_target is not None and track_ids[i] == self._last_track_target:
                            print(f'Agent {i+1} is the same as last track target')
                            last_leader_pred = self._last_pred['leader']
                            last_follower_pred = self._last_pred['follower']
                            self._last_belief['history_x'].append(agent_prediction[0, 0])
                            self._last_belief['history_y'].append(agent_prediction[1, 0])
                            agent_v = np.array([self._last_belief['history_x'][-1] - self._last_belief['history_x'][-2], self._last_belief['history_y'][-1] - self._last_belief['history_y'][-2]]) / DT
                            real_relative_v = np.dot(agent_v - ego_v, tangent_vector)
                            print(f"real_relative_v: {real_relative_v}")
                            ref_time = 5
                            sigma_leader = 2.0
                            sigma_follower = 2.0
                            rho = 0.25
                            # leader 
                            pred_v = np.array([last_leader_pred[0, ref_time] - last_leader_pred[0, 0], last_leader_pred[1, ref_time] - last_leader_pred[1, 0]]) / (ref_time * DT)
                            pred_relative_v = np.dot(pred_v - ego_v, tangent_vector)
                            # diff = real_relative_v - pred_relative_v
                            diff = real_relative_v - 1.0
                            # 正太分布根据diff计算观测概率

                            p_leader_ob = 1 / (np.sqrt(2 * np.pi) * sigma_leader) * np.exp(-(diff)**2 / (2 * sigma_leader**2)) 
                            print(f"leader_pred_relative_v: {pred_relative_v}")
                            print(f"leader_diff: {diff}")
                            print(f"p_leader_ob: {p_leader_ob}")
                            # follower
                            pred_v = np.array([last_follower_pred[0, ref_time] - last_follower_pred[0, 0], last_follower_pred[1, ref_time] - last_follower_pred[1, 0]]) / (ref_time * DT)
                            pred_relative_v = np.dot(pred_v - ego_v, tangent_vector)
                            # diff = real_relative_v - pred_relative_v
                            diff = real_relative_v - -1.0
                            # 正太分布根据diff计算观测概率
                            p_follower_ob = 1 / (np.sqrt(2 * np.pi) * sigma_follower) * np.exp(-(diff)**2 / (2 * sigma_follower**2))
                            k_ob_l_f = p_follower_ob / (p_leader_ob + p_follower_ob)
                            k_ob_f_l = p_leader_ob / (p_leader_ob + p_follower_ob)
                            print(f"follower_pred_relative_v: {pred_relative_v}")
                            print(f"follower_diff: {diff}")
                            print(f"p_follower_ob: {p_follower_ob}")
                            # calculate belief
                            p_leader = p_leader_ob * self._last_belief['leader'][-1] + k_ob_f_l * self._last_belief['follower'][-1] * 1 / (np.sqrt(2 * np.pi) * sigma_follower) * rho
                            p_follower = p_follower_ob * self._last_belief['follower'][-1] + k_ob_l_f * self._last_belief['leader'][-1] * 1 / (np.sqrt(2 * np.pi) * sigma_follower) * rho
                            sum_p = p_leader + p_follower
                            p_leader = p_leader / sum_p
                            p_follower = p_follower / sum_p
                            self._last_belief['leader'].append(p_leader)
                            self._last_belief['follower'].append(p_follower)

                        else:
                            print(f'Agent {i+1} is different from last track target')
                            self._last_track_target = track_ids[i]
                            self._last_belief = {"leader": [0.5], "follower": [0.5], "begin_iteration": iteration, "history_x": [agent_prediction[0, 0]], "history_y": [agent_prediction[1, 0]], "branch_time": []}
                            self._last_pred = {"leader": None, "follower": None}

                        # predict leader by s-t graph
                        agent_v = np.array([agent_prediction[0, 1] - agent_prediction[0, 0], agent_prediction[1, 1] - agent_prediction[1, 0]]) / DT
                        if len(self._last_belief['history_x']) > 1:
                            agent_v = np.array([self._last_belief['history_x'][-1] - self._last_belief['history_x'][-2], self._last_belief['history_y'][-1] - self._last_belief['history_y'][-2]]) / DT
                        x = agent_prediction[0, :]
                        y = agent_prediction[1, :]
                        path = CubicSpline2D(x=x, y=y)
                        # print(f"agent collsion t: {agent_collsion_t}")
                        # print(f"ego collsion t: {ego_collsion_t}")
                        # leader
                        target_t_ind = min(agent_collsion_t, ego_collsion_t)
                        l_pred_x, l_pred_y = LF_constant_prediction(True, agent_v, target_t_ind, path, target_speed, agent_prediction, T, DT, agent_collsion_t)
                        # follower
                        target_t_ind = max(agent_collsion_t, ego_collsion_t)
                        f_pred_x, f_pred_y = LF_constant_prediction(False, agent_v, target_t_ind, path, target_speed, agent_prediction, T, DT, agent_collsion_t)
                        # replace
                        global_predictions[i][0][0, 1:] = l_pred_x
                        global_predictions[i][0][1, 1:] = l_pred_y
                        self._last_pred['leader'] = copy.deepcopy(global_predictions[i][0][:2, :])
                        global_predictions[i][1][0, 1:] = f_pred_x
                        global_predictions[i][1][1, 1:] = f_pred_y
                        self._last_pred['follower'] = copy.deepcopy(global_predictions[i][1][:2, :])
                        belief = [self._last_belief['leader'][-1], self._last_belief['follower'][-1]]
                        track_dict[track_ids[i]] = -1
                        plt.cla()
                        t = np.arange(self._last_belief["begin_iteration"], iteration+1)
                        plt.plot(t, self._last_belief['leader'], 'r-')
                        plt.plot(t, self._last_belief['follower'], 'g--')
                        plt.xlabel('iteration')
                        plt.ylabel('belief')
                        plt.title('belief')
                        plt.legend(['leader', 'follower'])
                        save_belief_path = "/root/xzcllwx_ws/GameFormer-Planner/belief"
                        plt.savefig(os.path.join(save_belief_path, f'{iteration}.png'))
                        self._last_belief["branch_time"].append(calcualte_branch_time(self._last_belief['leader'][-1], 
                                                            self._last_pred['leader'],
                                                            self._last_belief['follower'][-1],
                                                            self._last_pred['follower'],
                                                            ego_v,
                                                            tangent_vector,
                                                            self._last_belief["branch_time"]))
                        if self.save_data:
                            self.frame_data['leader_belief'] = self._last_belief['leader']
                            self.frame_data['follower_belief'] = self._last_belief['follower']
                            self.frame_data['branch_time'] = self._last_belief["branch_time"]
                            self.frame_data['agent_history_x'] = self._last_belief['history_x']
                            self.frame_data['agent_history_y'] = self._last_belief['history_y']
                        break
                    else:
                        print(f'Agent {i+1} is not intersecting with global plan')
            if (len(belief) == 1):
                print(f'No Game: {belief[0]}')
                if self.save_data:
                    self.frame_data['game'] = False
            else:
                print(f'Leader: {belief[0]}')
                print(f'Follower: {belief[1]}')
                if self.save_data:
                    self.frame_data['game'] = True
                     
            plan, other_plan = self._confingency_planner.step(
                scenario=self._scenario,
                current_lanelet_id=0,
                time_step=iteration,
                ego_state=current_state,
                predictions=global_predictions,
                cov=cov,
                belief=belief,
                v_max=target_speed,
            )

            
            initial_condition = [
                                    ego_state.car_footprint.center.x, 
                                    ego_state.car_footprint.center.y,
                                    ego_state.car_footprint.center.heading,
                                    ego_state.dynamic_car_state.rear_axle_velocity_2d.x, 
                                ]
            modality_index = np.argmax(belief)
            new_plan = self._motion_planner.plan(
                initial_condition, 
                plan.astype(np.float64), 
                global_predictions[:, modality_index, :, :].astype(np.float64),
                target_speed,
                iteration
            )

            plt.cla()
            index_vector = np.arange(0, 80, 10)
            if f_pred_x is not None:
                if belief[0] >= belief[1]:
                    plt.plot(self._last_pred['leader'][0, index_vector], self._last_pred['leader'][1, index_vector], 'r^')
                    plt.plot(self._last_pred['follower'][0, index_vector], self._last_pred['follower'][1, index_vector], 'g^')
                else:
                    plt.plot(self._last_pred['leader'][0, index_vector], self._last_pred['leader'][1, index_vector], 'g^')
                    plt.plot(self._last_pred['follower'][0, index_vector], self._last_pred['follower'][1, index_vector], 'r^')
            plt.legend(['leader', 'follower'])
            plt.plot(plan[index_vector, 0], plan[index_vector, 1], 'ro')
            if other_plan is not None:
                other_len = other_plan.shape[0]
                other_index_vector = np.arange(0, other_len, 10)
                plt.plot(other_plan[other_index_vector, 0], other_plan[other_index_vector, 1], 'go')
            if self.save_data:
                if f_pred_x is not None:
                    self.frame_data['leader_x'] = (self._last_pred['leader'][0, :]).tolist()
                    self.frame_data['leader_y'] = (self._last_pred['leader'][1, :]).tolist()
                    self.frame_data['follower_x'] = (self._last_pred['follower'][0, :]).tolist()
                    self.frame_data['follower_y'] = (self._last_pred['follower'][1, :]).tolist()
                self.frame_data['origin_plan_x'] = (plan[:, 0]).tolist()
                self.frame_data['origin_plan_y'] = (plan[:, 1]).tolist()
                if other_plan is not None:
                    self.frame_data['other_plan_x'] = (other_plan[:, 0]).tolist()
                    self.frame_data['other_plan_y'] = (other_plan[:, 1]).tolist()

            plan = np.array(new_plan)
            plt.plot(plan[index_vector, 0], plan[index_vector, 1], 'r-')
            plt.axis('equal')
            save_contingency_path = "/root/xzcllwx_ws/GameFormer-Planner/contingency"
            plt.savefig(os.path.join(save_contingency_path, f'{iteration}.png'))
            if self.save_data:
                self.frame_data['new_paln_x'] = (plan[:, 0]).tolist()
                self.frame_data['new_paln_y'] = (plan[:, 1]).tolist() 
            # plt.show()
            plan[:, [2, 3]] = plan[:, [3, 2]] # [x, y, yaw, v]
            # ref_speed = [ target_speed for _ in plan[:, 3]]
            # ref_traj = [tuple(i) for i in plan[:, :3]]
            # self._smoother.set_reference_trajectory(
            #                                initial_condition,
            #                                ref_speed,
            #                                ref_traj,)  
            # self._smoother.set_obstacles(global_predictions[:,0,:,:])
            # solution = self._smoother.solve()
            # optimized_states = solution.value(self._smoother.state)
            # new_plan = []
            # for i in range(optimized_states.shape[1]):
            #     new_plan.append([
            #         optimized_states[0, i],
            #         optimized_states[1, i],
            #         optimized_states[2, i],
            #         optimized_states[3, i]
            #     ])
            # plan = np.array(new_plan)
            # Plot the original and optimized plans
            save_data_to_csv(self.frame_data, iteration, self.csv_save_dir)
            plan[:, 3] = calculate_path_curvature(plan) # [x, y, yaw, k]
            self._global_plan = plan
            plan = LatticePlanner.transform_to_ego_frame(plan, ego_state)
        
        # Trajectory refinement
        # with torch.no_grad():
        #     plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed, 
        #                                          predictions, plan, scores, ref_path, observation)
            
        states = transform_predictions_to_states(plan[:,:3], history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)

        return trajectory, plan, output_predictions, track_dict
        # return trajectory, plan, predictions[0].reshape(-1, T, D).cpu().numpy()
    
    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        if self._scenario_manager is None:
            self._scenario_manager = ScenarioManager(
                self._initialization.map_api, ego_state, self._initialization.route_roadblock_ids
            )
            self._initialize_route_plan(self._scenario_manager.get_route_roadblock_ids())
            self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, self._max_path_length)
            self._init_laneletnets(ego_state)
        trajectory, plan, predictions, track_dict = self._plan(iteration, ego_state, history, traffic_light_data, observation)

        self._render = True
        
        if self._render:
            self._imgs.append(
                    self._scene_render.render_from_simulation(
                        current_input=current_input,
                        initialization=self._initialization,
                        route_roadblock_ids=self._route_roadblock_ids,
                        iteration=current_input.iteration.index,
                        planning_trajectory=plan[:, :2],
                        predictions=predictions,
                        agent_attn_weights=track_dict,
                        return_img=self._render,
                    )
            )
            filename= f'{iteration}.png'
            img = self._imgs[-1]
            plt.imsave(os.path.join(self._save_dir, filename), img)
        else:
            self._scene_render.render_from_simulation(
                current_input=current_input,
                initialization=self._initialization,
                route_roadblock_ids=self._route_roadblock_ids,
                iteration=current_input.iteration.index,
                planning_trajectory=plan[:, :2],
                predictions=predictions,
                return_img=self._render,
            )

        print(f'Iteration {iteration}: {time.time() - s:.3f} s')

        return trajectory
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # 排除motion_planner对象
        state['_motion_planner'] = None
        state['_smoother'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 重新初始化motion_planner
        if self.config_path:
            self._motion_planner = motion_planning.motion_planner(self.config_path)
            self._smoother = MotionNonlinearSmoother(trajectory_len=T*10-1, dt=DT)
