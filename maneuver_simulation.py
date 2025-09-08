import numpy as np
from scipy.optimize import minimize
from satellite_function import lagrange_extrapolate, Time_window_of_danger_zone

# 地球引力常数 (km^3/s^2)
MU = 398600.4418

class Satellite:
    """存储单颗卫星的状态信息"""
    def __init__(self, sat_id, team, state_vector, is_maneuvering=False):
        """
        Args:
            sat_id (str): 卫星ID, e.g., 'blue_1'
            team (str): 阵营, 'red' or 'blue'
            state_vector (np.ndarray): 6维笛卡尔状态向量 [x, y, z, vx, vy, vz] (km, km/s)
            is_maneuvering (bool): 卫星是否在机动
        """
        self.sat_id = sat_id
        self.team = team
        self.state_vector = np.array(state_vector, dtype=float)
        self.is_maneuvering = is_maneuvering

    def update_state(self, new_state_vector):
        """更新卫星的状态向量"""
        self.state_vector = np.array(new_state_vector, dtype=float)

    @property
    def position(self):
        return self.state_vector[:3]

    @property
    def velocity(self):
        return self.state_vector[3:]

class ManeuverSimulation:
    """卫星机动与追逃博弈仿真环境"""
    def __init__(self):
        self.satellites = {}
        self.time = 0.0  # 仿真时间 (s)
        self.dt = 10.0   # 时间步长 (s)
        self.blue_maneuver_star = None
        self.red_protector_star = None
        self.red_main_star = None
        self.fly_around_start_time = -1

    def setup_initial_scene(self):
        """设置仿真初始场景"""
        # 1. 定义红方主星的初始轨道 (一个典型的LEO轨道)
        # a, e, i(rad), omega(rad), Omega(rad), f(rad)
        red_main_keplerian = np.array([6371 + 700, 0.001, np.deg2rad(51.6), 0.0, 0.0, 0.0])
        R0, V0 = Time_window_of_danger_zone.calculate_state_information(red_main_keplerian, miu=3.986e5)
        red_main_state_vector = np.hstack([R0, V0])
        self.red_main_star = Satellite('red_main', 'red', red_main_state_vector)
        self.satellites['red_main'] = self.red_main_star

        # 2. 创建其他卫星，初始相对距离50km
        # 为了简化，我们在主星的VNC坐标系下进行偏移来创建初始位置
        # 注意：这只是一个简化的初始布局，长期轨道演化会使其漂移
        r, v = self.red_main_star.position, self.red_main_star.velocity
        r_norm = r / np.linalg.norm(r)
        v_norm = v / np.linalg.norm(v)
        c_norm = np.cross(r_norm, v_norm)

        # 红方护卫星: 位于主星后方50km
        red_protector_pos = r - v_norm * 50
        red_protector_vel = v
        self.red_protector_star = Satellite('red_protector', 'red', np.hstack([red_protector_pos, red_protector_vel]))
        self.satellites['red_protector'] = self.red_protector_star

        # 蓝方5颗卫星: 分布在主星周围
        offsets = [
            r_norm * 50,      # 上方
            -r_norm * 50,     # 下方
            c_norm * 50,      # 左侧
            -c_norm * 50,     # 右侧
            v_norm * 50       # 前方
        ]
        for i in range(5):
            blue_pos = r + offsets[i]
            blue_vel = v # 简化处理，速度与主星相同
            blue_sat = Satellite(f'blue_{i+1}', 'blue', np.hstack([blue_pos, blue_vel]))
            self.satellites[f'blue_{i+1}'] = blue_sat
        
        print("场景初始化完成，卫星部署如下：")
        for name, sat in self.satellites.items():
            dist = np.linalg.norm(sat.position - self.red_main_star.position)
            print(f"- {name}: 距离主星 {dist:.2f} km")


    def _cost_function(self, dv_vectors, initial_state_A, initial_state_B, transfer_time):
        """Optimization cost function: total Delta-V and final distance error."""
        dv1 = dv_vectors[:3]
        dv2 = dv_vectors[3:]

        # State of satellite A after first impulse
        state_A_after_dv1 = initial_state_A.copy()
        state_A_after_dv1[3:] += dv1

        # Propagate A to the end of the transfer time
        final_state_A = lagrange_extrapolate(state_A_after_dv1, transfer_time)
        final_state_A[3:] += dv2 # Apply second impulse

        # Propagate B (target) to the end of the transfer time
        final_state_B = lagrange_extrapolate(initial_state_B, transfer_time)

        # Cost is the sum of Delta-V magnitude and a penalty for missing the target distance
        total_dv = np.linalg.norm(dv1) + np.linalg.norm(dv2)
        final_distance = np.linalg.norm(final_state_A[:3] - final_state_B[:3])
        
        # The penalty encourages the optimizer to meet the distance constraint
        distance_error = abs(final_distance - self.target_distance)
        
        return total_dv + distance_error * 100 # Penalty weight

    def plan_two_impulse_maneuver(self, sat_A, sat_B, target_distance, transfer_time=3600.0):
        """Plans an optimal two-impulse maneuver for sat_A to reach a target distance from sat_B."""
        print(f"\nPlanning maneuver for {sat_A.sat_id} to reach {target_distance}km from {sat_B.sat_id}...")
        self.target_distance = target_distance

        # Initial guess for the Delta-V vectors (small random values)
        initial_guess = np.random.rand(6) * 0.01 # km/s

        # Run the optimization
        result = minimize(
            self._cost_function,
            initial_guess,
            args=(sat_A.state_vector, sat_B.state_vector, transfer_time),
            method='SLSQP',
            options={'disp': True, 'maxiter': 100}
        )

        if result.success:
            dv1 = result.x[:3]
            dv2 = result.x[3:]
            print(f"Optimization successful!")
            print(f"  - Delta-V 1: {np.linalg.norm(dv1)*1000:.2f} m/s")
            print(f"  - Delta-V 2: {np.linalg.norm(dv2)*1000:.2f} m/s")
            print(f"  - Total Delta-V: {(np.linalg.norm(dv1) + np.linalg.norm(dv2))*1000:.2f} m/s")
            return dv1, dv2, transfer_time
        else:
            print("Optimization failed!")
            return None, None, None

    def run(self):
        """运行完整的仿真流程"""
        self.setup_initial_scene()

        # --- Step 1: Blue satellite maneuvers to 20km standoff ---
        # Randomly select a blue satellite to maneuver
        self.blue_maneuver_star = self.satellites[f'blue_{np.random.randint(1, 6)}']
        print(f"\n--- STEP 1: INITIATING MANEUVER ---")
        print(f"Selected {self.blue_maneuver_star.sat_id} to approach {self.red_main_star.sat_id}.")

        # Plan the maneuver
        dv1, dv2, transfer_time = self.plan_two_impulse_maneuver(
            self.blue_maneuver_star, 
            self.red_main_star, 
            target_distance=20.0, 
            transfer_time=3600.0 # Maneuver takes 1 hour
        )

        if dv1 is None:
            print("Maneuver planning failed. Aborting simulation.")
            return

        # Execute the maneuver over the simulation time
        # Apply first impulse
        self.blue_maneuver_star.state_vector[3:] += dv1
        self.blue_maneuver_star.is_maneuvering = True

        print(f"\nExecuting maneuver for {transfer_time} seconds...")
        maneuver_end_time = self.time + transfer_time
        while self.time < maneuver_end_time:
            # Propagate all satellites
            for sat in self.satellites.values():
                # The maneuvering satellite is already on its transfer orbit
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
            
            self.time += self.dt
            dist = np.linalg.norm(self.blue_maneuver_star.position - self.red_main_star.position)
            if int(self.time) % 300 == 0:
                 print(f"  [T_sim={self.time:.0f}s] Distance between {self.blue_maneuver_star.sat_id} and {self.red_main_star.sat_id}: {dist:.2f} km")

        # Apply second impulse
        self.blue_maneuver_star.state_vector[3:] += dv2
        self.blue_maneuver_star.is_maneuvering = False

        final_dist = np.linalg.norm(self.blue_maneuver_star.position - self.red_main_star.position)
        print(f"--- STEP 1: MANEUVER COMPLETE ---")
        print(f"Final distance to {self.red_main_star.sat_id}: {final_dist:.2f} km")

        # --- Station-Keeping (悬停：保持20km相对距离) ---
        print(f"\n--- STATION-KEEPING: {self.blue_maneuver_star.sat_id} 在 20km 处悬停 ---")
        station_keeping_duration = 1800.0  # 悬停30分钟，可按需调整
        station_keeping_end = self.time + station_keeping_duration
        TARGET_DIST = 20.0
        TOL = 0.1                 # 100 m 误差带
        CORRECT_ACC = 0.00001     # km/s^2 等效修正强度（简化连续微推力）

        while self.time < station_keeping_end:
            # 先推进其他卫星
            for sat in self.satellites.values():
                if sat is self.blue_maneuver_star:
                    continue
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))

            # 推进蓝方机动星一步
            new_state = lagrange_extrapolate(self.blue_maneuver_star.state_vector, self.dt)
            self.blue_maneuver_star.update_state(new_state)

            # 计算与红方主星的距离并做微小校正
            rel = self.blue_maneuver_star.position - self.red_main_star.position
            dist = np.linalg.norm(rel)
            err = dist - TARGET_DIST
            if abs(err) > TOL and dist > 1e-6:
                dir_unit = rel / dist
                # 误差为正：太远，向内修正；误差为负：太近，向外修正
                dv = -np.sign(err) * CORRECT_ACC * self.dt
                self.blue_maneuver_star.state_vector[3:] += dir_unit * dv

            self.time += self.dt
            if int(self.time) % 300 == 0:
                print(f"  [T_sim={self.time:.0f}s] Hover distance: {dist:.2f} km (err={err:.2f} km)")

        print("--- STATION-KEEPING COMPLETE ---")

        # --- Step 2: Red protector satellite maneuvers to 5km to fly-around ---
        print(f"\n--- STEP 2: INITIATING INTERCEPT ---")
        print(f"{self.red_protector_star.sat_id} is moving to intercept {self.blue_maneuver_star.sat_id}.")

        # Plan the intercept maneuver
        dv1_red, dv2_red, transfer_time_red = self.plan_two_impulse_maneuver(
            self.red_protector_star,
            self.blue_maneuver_star,
            target_distance=5.0,
            transfer_time=1800.0 # Intercept takes 30 minutes
        )

        if dv1_red is None:
            print("Intercept maneuver planning failed. Aborting simulation.")
            return
        
        # Execute the intercept maneuver
        # Apply first impulse
        self.red_protector_star.state_vector[3:] += dv1_red
        self.red_protector_star.is_maneuvering = True

        print(f"\nExecuting intercept for {transfer_time_red} seconds...")
        intercept_end_time = self.time + transfer_time_red
        while self.time < intercept_end_time:
            # Propagate all satellites
            for sat in self.satellites.values():
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
            
            self.time += self.dt
            dist = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
            if int(self.time) % 300 == 0:
                print(f"  [T_sim={self.time:.0f}s] Distance between {self.red_protector_star.sat_id} and {self.blue_maneuver_star.sat_id}: {dist:.2f} km")

        # Apply second impulse
        self.red_protector_star.state_vector[3:] += dv2_red
        self.red_protector_star.is_maneuvering = False

        final_dist_intercept = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
        print(f"--- STEP 2: INTERCEPT COMPLETE ---")
        print(f"Final distance to {self.blue_maneuver_star.sat_id}: {final_dist_intercept:.2f} km")
        print(f"{self.red_protector_star.sat_id} will now begin 'fly-around'.")

        self.fly_around_start_time = self.time

        # --- Step 3: Wait for 15 minutes, then start the escape maneuver ---
        print(f"\n--- STEP 3: FLY-AROUND & ESCAPE ---")
        print(f"Simulating 900s fly-around period before escape...")

        fly_around_end_time = self.time + 900.0
        while self.time < fly_around_end_time:
            for sat in self.satellites.values():
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
            self.time += self.dt

        print(f"\nFly-around time exceeded 15 minutes! {self.blue_maneuver_star.sat_id} is initiating escape.")

        # Pursuit-Evasion Game for 1 hour
        pursuit_duration = 3600.0
        pursuit_end_time = self.time + pursuit_duration
        # Define the magnitude of the continuous thrust (as impulse per time step)
        # Let's assume a small, constant 1 m/s burn per step for both
        EVASION_DV = 0.001 # km/s

        print(f"Running pursuit-evasion for {pursuit_duration} seconds...")
        while self.time < pursuit_end_time:
            # --- Calculate maneuvers for this time step ---
            pursuer = self.red_protector_star
            evader = self.blue_maneuver_star

            # Vector from pursuer to evader
            relative_pos = evader.position - pursuer.position
            direction = relative_pos / np.linalg.norm(relative_pos)

            # Evader burns away from the pursuer
            dv_evade = direction * EVASION_DV
            evader.state_vector[3:] += dv_evade

            # Pursuer burns towards the evader
            dv_pursue = direction * EVASION_DV
            pursuer.state_vector[3:] += dv_pursue

            # --- Propagate all satellites for the next time step ---
            for sat in self.satellites.values():
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
            
            self.time += self.dt

            # --- Log the distance ---
            dist = np.linalg.norm(relative_pos)
            if int(self.time) % 300 == 0:
                print(f"  [T_sim={self.time:.0f}s] Pursuit Distance: {dist:.2f} km")

        final_dist_pursuit = np.linalg.norm(self.blue_maneuver_star.position - self.red_protector_star.position)
        print(f"--- STEP 3: PURSUIT-EVASION COMPLETE ---")
        print(f"Final distance between pursuer and evader: {final_dist_pursuit:.2f} km")
        print("\nSIMULATION FINISHED.")

    def _execute_intercept_and_escape(self):
        # --- Step 2: Red protector satellite maneuvers to 5km to fly-around ---
        print(f"\n--- STEP 2: INITIATING INTERCEPT ---")
        print(f"{self.red_protector_star.sat_id} is moving to intercept {self.blue_maneuver_star.sat_id}.")

        # Plan the intercept maneuver
        dv1_red, dv2_red, transfer_time_red = self.plan_two_impulse_maneuver(
            self.red_protector_star,
            self.blue_maneuver_star,
            target_distance=5.0,
            transfer_time=1800.0 # Intercept takes 30 minutes
        )

        if dv1_red is None:
            print("Intercept maneuver planning failed. Aborting simulation.")
            return
        
        # Execute the intercept maneuver
        self.red_protector_star.state_vector[3:] += dv1_red
        self.red_protector_star.is_maneuvering = True

        print(f"\nExecuting intercept for {transfer_time_red} seconds...")
        intercept_end_time = self.time + transfer_time_red
        while self.time < intercept_end_time:
            for sat in self.satellites.values():
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
            self.time += self.dt

        self.red_protector_star.state_vector[3:] += dv2_red
        self.red_protector_star.is_maneuvering = False

        final_dist_intercept = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
        print(f"--- STEP 2: INTERCEPT COMPLETE ---")
        print(f"Final distance to {self.blue_maneuver_star.sat_id}: {final_dist_intercept:.2f} km")

        # --- Fly-Around Phase (绕飞) for 15 minutes ---
        print(f"\n--- INITIATING FLY-AROUND ---")
        fly_around_duration = 900.0 # 15 minutes
        fly_around_end_time = self.time + fly_around_duration
        self.fly_around_start_time = self.time

        while self.time < fly_around_end_time:
            # Propagate all non-maneuvering satellites
            for sat in self.satellites.values():
                if sat not in [self.red_protector_star, self.blue_maneuver_star]:
                    sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
            
            # Blue star continues station-keeping
            # (For simplicity, we assume it just propagates for this phase)
            self.blue_maneuver_star.update_state(lagrange_extrapolate(self.blue_maneuver_star.state_vector, self.dt))

            # Red protector performs fly-around
            # This is a simplified implementation of a forced circular relative orbit
            relative_pos = self.red_protector_star.position - self.blue_maneuver_star.position
            relative_vel = self.red_protector_star.velocity - self.blue_maneuver_star.velocity
            dist = np.linalg.norm(relative_pos)
            
            # Required velocity for a circular orbit at this distance and relative speed
            # v_circ = sqrt(mu / r), here we use a simplified relative gravity
            # We need a continuous acceleration to keep it in a circle
            radial_dir = relative_pos / dist
            tangential_dir = np.cross(radial_dir, [0,0,1]) # Simplified tangential direction
            tangential_dir /= np.linalg.norm(tangential_dir)

            # Acceleration to maintain circular path and correct distance
            # This is a highly simplified control logic
            accel_radial = - (np.dot(relative_vel, relative_vel) / dist) * radial_dir
            accel_dist_correction = -0.00001 * (dist - 5.0) * radial_dir
            total_accel = accel_radial + accel_dist_correction

            self.red_protector_star.state_vector[3:] += total_accel * self.dt
            self.red_protector_star.update_state(lagrange_extrapolate(self.red_protector_star.state_vector, self.dt))

            self.time += self.dt
            if int(self.time) % 300 == 0:
                print(f"  [T_sim={self.time:.0f}s] Fly-around. Current distance: {dist:.2f} km")

        # --- Step 3: Start the escape maneuver ---
        print(f"\nFly-around time exceeded 15 minutes! {self.blue_maneuver_star.sat_id} is initiating escape.")

        # Pursuit-Evasion Game for 1 hour
        pursuit_duration = 3600.0
        pursuit_end_time = self.time + pursuit_duration
        EVASION_DV_STEP = 0.001 # km/s per step

        print(f"Running pursuit-evasion for {pursuit_duration} seconds...")
        while self.time < pursuit_end_time:
            pursuer = self.red_protector_star
            evader = self.blue_maneuver_star

            relative_pos = evader.position - pursuer.position
            direction = relative_pos / np.linalg.norm(relative_pos)

            evader.state_vector[3:] += direction * EVASION_DV_STEP
            pursuer.state_vector[3:] += direction * EVASION_DV_STEP

            for sat in self.satellites.values():
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
            
            self.time += self.dt

            dist = np.linalg.norm(relative_pos)
            if int(self.time) % 600 == 0:
                print(f"  [T_sim={self.time:.0f}s] Pursuit Distance: {dist:.2f} km")

        final_dist_pursuit = np.linalg.norm(self.blue_maneuver_star.position - self.red_protector_star.position)
        print(f"--- PURSUIT-EVASION COMPLETE ---")
        print(f"Final distance between pursuer and evader: {final_dist_pursuit:.2f} km")
        print("\nSIMULATION FINISHED.")

if __name__ == '__main__':
    simulation = ManeuverSimulation()
    simulation.run()

