from typing import Any, Dict, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from casadi import DM, Opti, OptiSol, cos, diff, sin, sumsqr, vertcat
Pose = Tuple[float, float, float]  # (x, y, yaw)


class MotionNonlinearSmoother:
    """
    Smoothing a set of xy observations with a vehicle dynamics model.
    Solved with direct multiple-shooting.
    :param trajectory_len: trajectory length
    :param dt: timestep (sec)
    """

    def __init__(self, trajectory_len: int, dt: float):
        """
        :param trajectory_len: the length of trajectory to be optimized.
        :param dt: the time interval between trajectory points.
        """
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.current_index = 0
        # Use a array of dts to make it compatible to situations with varying dts across different time steps.
        self._dts: npt.NDArray[np.float32] = np.asarray([[dt] * trajectory_len])
        self._collision_constraints = []  # 存储碰撞约束引用
        self.vehicle_length = 5.0
        self.vehicle_width = 2.1
        self.safety_margin = 0.5
        
        self.vehicle_circles = [
            (0.0, 0.0, 1.2),              # (偏移x, 偏移y, 半径)
            (self.vehicle_length/4, 0, 1.0),  # 前部
            (-self.vehicle_length/4, 0, 1.0)  # 后部
        ]
        
        self._init_optimization()

    def _init_optimization(self) -> None:
        """
        Initialize related variables and constraints for optimization.
        """
        self.nx = 4  # state dim
        self.nu = 2  # control dim

        self._optimizer = Opti()  # Optimization problem
        self._create_decision_variables()
        self._create_parameters()
        self._set_dynamic_constraints()
        self._set_state_constraints()
        self._set_control_constraints()
        self._set_objective()
        
        self.constraint_active = self._optimizer.parameter(1, 1)
        self._optimizer.set_value(self.constraint_active, 1)  # 默认激活

        self._add_collision_constraints()
        # Set default solver options (quiet)
        options = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.max_iter": 30}
        self.set_solver_optimizerons(options)

    def set_reference_trajectory(self, x_curr: Sequence[float], ref_speed: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """
        Set the reference trajectory that the smoother is trying to loosely track.
        :param x_curr: current state of size nx (x, y, yaw, speed)
        :param reference_trajectory: N+1 x 3 reference, where the second dim is for (x, y, yaw)
        """
        self._check_inputs(x_curr, reference_trajectory)

        self._optimizer.set_value(self.x_curr, DM(x_curr))
        self._optimizer.set_value(self.ref_traj, DM(reference_trajectory).T)
        self._set_initial_guess(ref_speed, reference_trajectory)

    def set_solver_optimizerons(self, options: Dict[str, Any]) -> None:
        """
        Control solver options including verbosity.
        :param options: Dictionary containing optimization criterias
        """
        self._optimizer.solver("ipopt", options)

    def solve(self) -> OptiSol:
        """
        Solve the optimization problem. Assumes the reference trajectory was already set.
        :return Casadi optimization class
        """

        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
        """
        Define the decision variables for the trajectory optimization.
        """
        # State trajectory (x, y, yaw, speed)
        self.state = self._optimizer.variable(self.nx, self.trajectory_len + 1)
        self.position_x = self.state[0, :]
        self.position_y = self.state[1, :]
        self.yaw = self.state[2, :]
        self.speed = self.state[3, :]

        # Control trajectory (curvature, accel)
        self.control = self._optimizer.variable(self.nu, self.trajectory_len)
        self.curvature = self.control[0, :]
        self.accel = self.control[1, :]

        # Derived control and state variables, dt[:, 1:] becuases state vector is one step longer than action.
        self.curvature_rate = diff(self.curvature) / self._dts[:, 1:]
        self.jerk = diff(self.accel) / self._dts[:, 1:]
        self.lateral_accel = self.speed[:self.trajectory_len] ** 2 * self.curvature

    def _create_parameters(self) -> None:
        """
        Define the expert trjactory and current position for the trajectory optimizaiton.
        """
        self.ref_traj = self._optimizer.parameter(3, self.trajectory_len + 1)  # (x, y, yaw)
        self.x_curr = self._optimizer.parameter(self.nx, 1)
        
        # 修改参数结构以支持椭圆和预测轨迹
        # 时间步 x 参数 x 障碍物数量
        self.max_obstacles = 5 
        # 参数: [x, y, heading, a, b, confidence]
        # a: 长半轴, b: 短半轴
        self.obstacle_params = self._optimizer.parameter(5, self.max_obstacles * (self.trajectory_len + 1))
        self.n_obstacles = self._optimizer.parameter(1, 1)  # 实际障碍物数量

    def _compute_ellipse_distance(self, point_x, point_y, ellipse_x, ellipse_y, heading, a, b):
        """计算点到椭圆的距离，增强数值稳定性"""
        from casadi import cos, sin, sqrt, fmax, fmin, if_else, atan2
        
        # 处理半轴可能为零的情况
        a_safe = fmax(a, 0.1)  # 确保半轴至少为0.1m
        b_safe = fmax(b, 0.1)
        
        # 将点转到椭圆坐标系
        dx = point_x - ellipse_x
        dy = point_y - ellipse_y
        
        # 旋转到椭圆主轴方向
        cos_h = cos(heading)
        sin_h = sin(heading)
        x_rot = cos_h * dx + sin_h * dy
        y_rot = -sin_h * dx + cos_h * dy
        
        # 归一化坐标
        x_norm = x_rot / a_safe
        y_norm = y_rot / b_safe
        
        # 计算归一化坐标系中点到单位圆的距离
        norm_sq = x_norm**2 + y_norm**2
        norm_sq_safe = fmax(norm_sq, 1e-10)  # 确保不接近零
        
        # 使用数值更稳定的计算方法
        dist = if_else(
            norm_sq <= 1,
            # 内部点: 使用更稳定的公式计算负距离
            -fmin(a_safe, b_safe) * (1.0 - sqrt(norm_sq_safe)),
            # 外部点: 更安全的计算
            sqrt(
                (x_rot)**2 + (y_rot)**2
            ) - sqrt(
                (a_safe * cos(atan2(y_norm, x_norm)))**2 + 
                (b_safe * sin(atan2(y_norm, x_norm)))**2
            )
        )
        
        return dist
    
    def _set_dynamic_constraints(self) -> None:
        r"""
        Set the system dynamics constraints as following:
          dx/dt = f(x,u)
          \dot{x} = speed * cos(yaw)
          \dot{y} = speed * sin(yaw)
          \dot{yaw} = speed * curvature
          \dot{speed} = accel
        """
        state = self.state
        control = self.control
        dt = self.dt

        def process(x: Sequence[float], u: Sequence[float]) -> Any:
            """Process for state propagation."""
            return vertcat(x[3] * cos(x[2]), x[3] * sin(x[2]), x[3] * u[0], u[1])

        for k in range(self.trajectory_len):  # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = process(state[:, k], control[:, k])
            k2 = process(state[:, k] + dt / 2 * k1, control[:, k])
            k3 = process(state[:, k] + dt / 2 * k2, control[:, k])
            k4 = process(state[:, k] + dt * k3, control[:, k])
            next_state = state[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self._optimizer.subject_to(state[:, k + 1] == next_state)  # close the gaps

    def _set_control_constraints(self) -> None:
        """Set the hard control constraints."""
        curvature_limit = 1.0 / 3.0  # 1/m
        self._optimizer.subject_to(self._optimizer.bounded(-curvature_limit, self.curvature, curvature_limit))

        accel_limit = 2.4  # m/s^2
        decel_limit = -4.0  # m/s^2
        self._optimizer.subject_to(self._optimizer.bounded(decel_limit, self.accel, accel_limit))

    def _set_state_constraints(self) -> None:
        """Set the hard state constraints."""
        # Constrain the current time -- initial boundary condition
        self._optimizer.subject_to(self.state[:, self.current_index] == self.x_curr)

        max_speed = 20.0  # m/s
        self._optimizer.subject_to(self._optimizer.bounded(0.0, self.speed, max_speed))  # only forward

    def _set_objective(self) -> None:
        """Set the objective function. Use care when modifying these weights."""
        # Follow reference, minimize control rates and absolute inputs
        alpha_xy = 1.0
        alpha_yaw = 10.0
        alpha_rate = 0.6
        alpha_abs = 0.1
        alpha_lat_accel = 0.3
        cost_stage = (
            alpha_xy * sumsqr(self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y))
            + alpha_yaw * sumsqr(self.ref_traj[2, :] - self.yaw)
            + alpha_rate * (sumsqr(self.curvature_rate) + sumsqr(self.jerk))
            + alpha_abs * (sumsqr(self.curvature) + sumsqr(self.accel))
            + alpha_lat_accel * sumsqr(self.lateral_accel)
        )

        self._optimizer.minimize(cost_stage)
    
    def set_obstacles(self, obstacles_trajectories: np.ndarray) -> None:
        """设置带有预测轨迹的障碍物信息"""
        n_obstacles = min(obstacles_trajectories.shape[0], self.max_obstacles)
        obstacle_data = np.zeros((5, self.max_obstacles * (self.trajectory_len + 1)))
        
        # 最小半轴长度以确保数值稳定性
        min_axis_length = 0.1
        
        # 填充障碍物预测轨迹
        for i in range(n_obstacles):
            traj = obstacles_trajectories[i]
            
            # 检查预测轨迹长度
            pred_steps = min(traj.shape[1], self.trajectory_len + 1)
            
            # 填充已知的时间步
            for t in range(pred_steps):
                offset = t * self.max_obstacles + i
                
                # 基本状态
                obstacle_data[0, offset] = traj[0, t]  # x
                obstacle_data[1, offset] = traj[1, t]  # y
                obstacle_data[2, offset] = traj[2, t]  # heading
                
                # 动态调整椭圆大小 - 确保数值稳定性
                if traj.shape[0] >= 5:
                    obstacle_data[3, offset] = max(min_axis_length, traj[3, t])  # a
                    obstacle_data[4, offset] = max(min_axis_length, traj[4, t])  # b
                else:
                    obstacle_data[3, offset] = 3.0  # 默认长半轴
                    obstacle_data[4, offset] = 1.5  # 默认短半轴
            
            # 填充剩余时间步...
        
        # 更新优化器参数
        self._optimizer.set_value(self.obstacle_params, DM(obstacle_data))
        self._optimizer.set_value(self.n_obstacles, n_obstacles)
        self._optimizer.set_value(self.constraint_active, 1 if n_obstacles > 0 else 0)
    
    def _add_collision_constraints(self) -> None:
        """添加参数化的碰撞约束"""
        from casadi import if_else, sqrt, cos, sin
        
        # 为规划轨迹的每个点添加碰撞约束
        for k in range(self.trajectory_len + 1):
            x_pos = self.position_x[k]
            y_pos = self.position_y[k]
            yaw = self.yaw[k]
            
            # 应用车辆多圆模型
            for circle_idx, (circle_dx, circle_dy, circle_r) in enumerate(self.vehicle_circles):
                # 计算全局坐标中的圆心位置
                circle_x = x_pos + circle_dx * cos(yaw) - circle_dy * sin(yaw)
                circle_y = y_pos + circle_dx * sin(yaw) + circle_dy * cos(yaw)
                
                # 对当前时间步的每个障碍物
                for i in range(self.max_obstacles):
                    # 计算障碍物参数在平坦数组中的索引
                    obs_idx = k * self.max_obstacles + i
                    
                    # 获取障碍物参数
                    obs_x = self.obstacle_params[0, obs_idx]
                    obs_y = self.obstacle_params[1, obs_idx]
                    obs_heading = self.obstacle_params[2, obs_idx]
                    obs_a = self.obstacle_params[3, obs_idx]
                    obs_b = self.obstacle_params[4, obs_idx]
                    
                    # 计算到椭圆的距离
                    distance = self._compute_ellipse_distance(
                        circle_x, circle_y, obs_x, obs_y, obs_heading, obs_a, obs_b
                    )
                    
                    # 计算安全距离
                    safety_margin = self.safety_margin
                    
                    # 使用条件约束，只对有效障碍物生效
                    # i < n_obstacles && constraint_active == 1.0
                    condition = (i < self.n_obstacles) * (self.constraint_active > 0.5)
                    
                    self._optimizer.subject_to(
                        if_else(
                            condition,
                            distance - (circle_r + safety_margin),
                            1.0  # 默认满足
                        ) >= 0
                    )

    def _set_initial_guess(self, ref_speed: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """Set a warm-start for the solver based on the reference trajectory."""
        # Initialize state guess based on reference
        self._optimizer.set_initial(self.state[:3, :], DM(reference_trajectory).T)  # (x, y, yaw)
        self._optimizer.set_initial(self.state[3, :], DM(ref_speed))  # speed

    def _check_inputs(self, x_curr: Sequence[float], reference_trajectory: Sequence[Pose]) -> None:
        """Raise ValueError if inputs are not of proper size."""
        if len(x_curr) != self.nx:
            raise ValueError(
                f"x_curr length {len(x_curr)} must be equal to state dim {self.nx}")

        if len(reference_trajectory) != self.trajectory_len + 1:
            raise ValueError(
                f"reference traj length {len(reference_trajectory)} must be equal to {self.trajectory_len + 1}"
            )
