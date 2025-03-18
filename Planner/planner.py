import math
import time
import matplotlib.pyplot as plt
from shapely import Point, LineString
from .planner_utils import *
from .observation import *
from GameFormer.predictor import GameFormer
from .state_lattice_path_planner import LatticePlanner

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring

import sys
import os
import copy
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
        # load settings from planning_fast.json
        settings_dict = load_planning_json("planning_fast.json")
        settings_dict["risk_dict"] = risk_dict = load_risk_json()
        if not self._time:
            settings_dict["evaluation_settings"]["show_visualization"] = True 
        self.settings = settings_dict
        self.vehicle_params = VehicleParameters(settings_dict["evaluation_settings"]["vehicle_type"])
        self.exec_timer = ExecTimer(timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"])
        self.frenet_settings = settings_dict["frenet_settings"]
        
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
        self._initialize_model()
        self._confingency_planner = None
        self._trajectory_planner = TrajectoryPlanner()
        self._path_planner = None
        self._scenario_manager = None
        self._scene_render = NuplanScenarioRender()
        self._imgs = []
        self._save_dir = '/root/xzcllwx_ws/GameFormer-Planner/figure'
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

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
        self._N_agent = 10
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
            ref_path = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        except:
            ref_path = None

        if ref_path is None:
            return None

        # Annotate red light to occupancy
        occupancy = np.zeros(shape=(ref_path.shape[0], 1))
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                conn_path = lane_conn.baseline_path.discrete_path
                conn_path = np.array([[p.x, p.y] for p in conn_path])
                red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

        # Annotate max speed along the reference path
        target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]
        if len(ref_path) < MAX_LEN * 10:
            ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10-len(ref_path), axis=0), axis=0)
        
        return ref_path.astype(np.float32)

    def _get_prediction(self, features):
        predictions, plan = self._model(features)
        K = len(predictions) // 2 - 1
        final_predictions = predictions[f'level_{K}_interactions'][:, 1:]
        final_scores = predictions[f'level_{K}_scores']
        ego_current = features['ego_agent_past'][:, -1]
        neighbors_current = features['neighbor_agents_past'][:, :, -1]
        final_scores = torch.nn.functional.softmax(final_scores, dim=-1)

        return plan, final_predictions, final_scores, ego_current, neighbors_current
    
    def _plan(self, ego_state, history, traffic_light_data, observation):
        # Construct input features
        # 转换到了ego坐标系下
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device) 

        # Get reference path
        ref_path = self._get_reference_path(ego_state, traffic_light_data, observation)

        # Infer prediction model
        with torch.no_grad():
            plan, predictions, scores, ego_state_transformed, neighbors_state_transformed = self._get_prediction(features)

        # Trajectory refinement
        with torch.no_grad():
            plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed, 
                                                 predictions, plan, scores, ref_path, observation)
            
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)
        
        _, N, _, _, _ = predictions.shape
        scores = scores.squeeze(0)  # 移除批次维度，变成 [N, M]
        scores =  scores[1:, :]  # 移除第一个agent的预测
        predictions = predictions.squeeze(0)  # 移除批次维度，变成 [N, M, T, D]
        
        best_indices = torch.argmax(scores, dim=1)  # 维度 [N]
        agent_indices = torch.arange(N)  # 维度 [N]

        best_trajectories = predictions[agent_indices, best_indices]  # 维度 [N, T, D]

        return trajectory, plan, best_trajectories.cpu().numpy(), ref_path
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
        trajectory, plan, predictions, ref_path = self._plan(ego_state, history, traffic_light_data, observation)
        
        state_args = dict()
        state_args['position'] = np.array([ego_state.car_footprint.center.x, ego_state.car_footprint.center.y])
        state_args['orientation'] = ego_state.car_footprint.center.heading
        state_args['velocity'] = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        state_args['acceleration'] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
        state_args['yaw_rate'] = ego_state.dynamic_car_state.angular_velocity
        # state_args['slip_angle'] = ego_state.dynamic_car_state.tire_steering_rate
        state_args['slip_angle'] = None
        state_args['time_step'] = current_input.iteration.index
        current_state = State(**state_args)
        self._confingency_planner.step(
            scenario=self._scenario,
            current_lanelet_id=0,
            time_step=current_input.iteration.index,
            ego_state=current_state,
            prediction=copy.deepcopy(predictions[:self._N_agent]),
            ref_path=copy.deepcopy(ref_path[:,:2]),
        )
        
        self._render = False
        
        if self._render:
            self._imgs.append(
                    self._scene_render.render_from_simulation(
                        current_input=current_input,
                        initialization=self._initialization,
                        route_roadblock_ids=self._route_roadblock_ids,
                        iteration=current_input.iteration.index,
                        planning_trajectory=plan[:, :2],
                        predictions=predictions,
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
