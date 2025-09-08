
import numpy as np
from scipy.optimize import minimize, fsolve
from satellite_function import Time_window_of_danger_zone



class SimConfig:
    dt = 10.0 # 时间步长 (s)
    log_file = 'maneuver_log.txt'
    # (km^3/s^2)
    MU = 398600.4418
    # (m^3/s^2)
    # MU = 986e14

    # Step1 悬停（20 km）控制参数
    station_target_dist = 20.0
    station_duration = 1800.0
    station_tol_km = 0.1
    station_kp = 1.0e-3
    station_kv = 1.0e-1
    station_ki = 1.0e-5
    station_dv_step_max = 0.0025  # km/s
    station_dead_err = 0.2        # km
    station_dead_vr = 0.01        # km/s
    station_err_int_max = 500.0   # km*s

    # Step2 两脉冲拦截（默认入口参数）
    intercept_transfer_time = 1800.0

    # Step2 相对PD收敛参数
    pd_duration = 1200.0
    pd_kpr = 2e-3
    pd_kv = 1e-1
    pd_dv_max = 0.002

    # 绕飞控制（5 km 目标半径）
    fly_time = 1200.0
    fly_period_hours = 24.0     # 目标角速度由此计算
    fly_kpr = 5.0e-3
    fly_kv_r = 2.0e-1
    fly_kpt = 1e-4
    fly_dv_max = 0.0010
    fly_gate_err = 0.2          # 进入切向控制前的半径误差阈值

    # 初始轨道与部署
    # 红主星六根数: a(km), e, i(rad), omega(rad), Omega(rad), f(rad)
    red_main_keplerian = np.array([6971 + 700, 0.001, np.deg2rad(91.6), 0.0, 0.0, 0.0])
    # 初始相对部署距离（红护卫与5颗蓝星相对红主星的初始偏移模长，km）
    initial_relative_distance_km = 50.0

class Satellite:
    """卫星状态信息"""
    def __init__(self, sat_id, team, state_vector, is_maneuvering=False):
        """
        Args:
            sat_id (str): 卫星ID, e.g., 'blue_1'
            team (str): 阵营, 'red' or 'blue'
            state_vector (np.ndarray): 笛卡尔状态向量 [x, y, z, vx, vy, vz] (km, km/s)
            is_maneuvering (bool): 卫星是否在机动
        """
        self.sat_id = sat_id
        self.team = team
        self.state_vector = np.array(state_vector, dtype=float)
        self.is_maneuvering = is_maneuvering

    def update_state(self, new_state_vector):
        self.state_vector = np.array(new_state_vector, dtype=float)

    @property
    def position(self):
        return self.state_vector[:3]

    @property
    def velocity(self):
        return self.state_vector[3:]


def lagrange_extrapolate(state_vector, dt, mu=SimConfig.MU):
    """
    拉格朗日外推（km, km/s, s）。
    """
    R0 = state_vector[:3]
    V0 = state_vector[3:]
    elements = Time_window_of_danger_zone.calculate_orbital_elements(mu, R0, V0)
    # 仅处理椭圆/双曲线（len==6）与圆（len==4）
    if len(elements) == 6:
        a, e, i, omega, Omega, f = elements
        r = a * (1 - e ** 2) / (1 + e * np.cos(f))
    elif len(elements) == 4:
        a, i, u, Omega = elements
        e = 0.0
        r = a
    else:
        # 抛物线等少见情况，简单返回原状态（可按需扩展）
        return state_vector.copy()

    sigma = np.dot(R0, V0) / np.sqrt(mu)

    def delta_E_equation(delta_E):
        return delta_E + sigma * (1 - np.cos(delta_E)) / np.sqrt(a) \
               - (1 - r / a) * np.sin(delta_E) - np.sqrt(mu / (a ** 3)) * dt

    # 求解偏近点角差
    delta_E = fsolve(delta_E_equation, 0.0)[0]

    # 拉格朗日系数
    r_next = a + (r - a) * np.cos(delta_E) + sigma * np.sqrt(a) * np.sin(delta_E)
    F = 1 - a * (1 - np.cos(delta_E)) / r
    G = a * sigma * (1 - np.cos(delta_E)) / np.sqrt(mu) + r * np.sqrt(a / mu) * np.sin(delta_E)
    F_t = -np.sqrt(mu * a) * np.sin(delta_E) / (r_next * r)
    G_t = 1 - a * (1 - np.cos(delta_E)) / r_next

    R = F * R0 + G * V0
    V = F_t * R0 + G_t * V0
    return np.hstack([R, V])


class ManeuverSimulation:
    """卫星机动与追逃博弈仿真环境"""

    def __init__(self):
        self.satellites = {}
        self.time = 0.0  # 仿真时间 (s)
        self.dt = SimConfig.dt  # 时间步长 (s)
        self.blue_maneuver_star = None
        self.red_protector_star = None
        self.red_main_star = None
        self.fly_around_start_time = -1
        self.log_f = open(SimConfig.log_file, 'w', encoding='utf-8')

    def _elements_from_state(self, state_vector):
        """从笛卡尔状态获取六根数，角度以度输出。
        返回: (a, e, i_deg, omega_deg, Omega_deg, f_deg)
        """
        R = state_vector[:3]
        V = state_vector[3:]
        elems = Time_window_of_danger_zone.calculate_orbital_elements(SimConfig.MU, R, V)
        if len(elems) == 6:
            a, e, i, omega, Omega, f = elems
        elif len(elems) == 4:
            a, i, u, Omega = elems
            e = 0.0
            omega = 0.0
            f = u
        else:
            a = float('nan'); e = float('nan'); i = 0.0; omega = 0.0; Omega = 0.0; f = 0.0
        return a, e, np.rad2deg(i), np.rad2deg(omega), np.rad2deg(Omega), np.rad2deg(f)

    def _log(self, text):
        try:
            self.log_f.write(text + "\n")
        except Exception:
            pass

    def _log_satellite_states(self, stage_tag):
        """将当前时刻所有卫星的状态、六根数与相对距离写入日志"""
        self._log(f"[T={self.time:.1f}s][{stage_tag}] 共 {len(self.satellites)} 颗卫星")
        red_main_pos = self.red_main_star.position if self.red_main_star is not None else None
        blue_target = self.blue_maneuver_star
        red_prot = self.red_protector_star
        for name, sat in self.satellites.items():
            x, y, z = sat.position
            vx, vy, vz = sat.velocity
            a, e, i_deg, omg_deg, OMG_deg, f_deg = self._elements_from_state(sat.state_vector)
            # 相对距离：到红主星、到蓝机动星、到红护卫星
            d_red = np.linalg.norm(sat.position - red_main_pos) if red_main_pos is not None else float('nan')
            d_blue = np.linalg.norm(sat.position - blue_target.position) if blue_target is not None else float('nan')
            d_rprot = np.linalg.norm(sat.position - red_prot.position) if red_prot is not None else float('nan')
            self._log(
                f"  - {name}: r=({x:.3f}, {y:.3f}, {z:.3f}) km, v=({vx:.6f}, {vy:.6f}, {vz:.6f}) km/s, "
                f"kep[a={a:.3f} km, e={e:.6f}, i={i_deg:.3f} deg, omega={omg_deg:.3f} deg, Omega={OMG_deg:.3f} deg, f={f_deg:.3f} deg], "
                f"dist[red_main={d_red:.3f} km, to_blue={d_blue:.3f} km, to_red_prot={d_rprot:.3f} km]"
            )

    def setup_initial_scene(self):
        """设置仿真初始场景"""
        # 1. 定义红方主星的初始轨道（来自配置）
        red_main_keplerian = SimConfig.red_main_keplerian
        R0, V0 = Time_window_of_danger_zone.calculate_state_information(red_main_keplerian, miu=3.986e5)
        red_main_state_vector = np.hstack([R0, V0])
        self.red_main_star = Satellite('red_main', 'red', red_main_state_vector)
        self.satellites['red_main'] = self.red_main_star

        # 2. 创建其他卫星，初始相对距离50km
        # 为了简化，我们在主星的VNC坐标系下进行偏移来创建初始位置
        r, v = self.red_main_star.position, self.red_main_star.velocity
        r_norm = r / np.linalg.norm(r)
        v_norm = v / np.linalg.norm(v)
        c_norm = np.cross(r_norm, v_norm)

        # 红方护卫星: 位于主星后方 initial_relative_distance_km
        red_protector_pos = r - v_norm * SimConfig.initial_relative_distance_km
        red_protector_vel = v
        self.red_protector_star = Satellite('red_protector', 'red', np.hstack([red_protector_pos, red_protector_vel]))
        self.satellites['red_protector'] = self.red_protector_star

        # 蓝方5颗卫星: 分布在主星周围（模长为 initial_relative_distance_km）
        offsets = [
            r_norm * SimConfig.initial_relative_distance_km,  # 上方
            -r_norm * SimConfig.initial_relative_distance_km,  # 下方
            c_norm * SimConfig.initial_relative_distance_km,  # 左侧
            -c_norm * SimConfig.initial_relative_distance_km,  # 右侧
            v_norm * SimConfig.initial_relative_distance_km  # 前方
        ]
        for i in range(5):
            blue_pos = r + offsets[i]
            blue_vel = v  # 简化处理，速度与主星相同
            blue_sat = Satellite(f'blue_{i + 1}', 'blue', np.hstack([blue_pos, blue_vel]))
            self.satellites[f'blue_{i + 1}'] = blue_sat

        print("场景初始化完成，卫星部署如下：")
        for name, sat in self.satellites.items():
            dist = np.linalg.norm(sat.position - self.red_main_star.position)
            print(f"- {name}: 距离主星 {dist:.2f} km")
        # 记录初始化
        self._log_satellite_states("INIT")

    def _cost_function(self, dv_vectors, initial_state_A, initial_state_B, transfer_time):
        """Optimization cost function: total Delta-V and final distance error."""
        dv1 = dv_vectors[:3]
        dv2 = dv_vectors[3:]

        # State of satellite A after first impulse
        state_A_after_dv1 = initial_state_A.copy()
        state_A_after_dv1[3:] += dv1

        # Propagate A to the end of the transfer time
        final_state_A = lagrange_extrapolate(state_A_after_dv1, transfer_time)
        final_state_A[3:] += dv2  # Apply second impulse

        # Propagate B (target) to the end of the transfer time
        final_state_B = lagrange_extrapolate(initial_state_B, transfer_time)

        # Cost is the sum of Delta-V magnitude and a penalty for missing the target distance
        total_dv = np.linalg.norm(dv1) + np.linalg.norm(dv2)
        final_distance = np.linalg.norm(final_state_A[:3] - final_state_B[:3])

        # The penalty encourages the optimizer to meet the distance constraint
        distance_error = abs(final_distance - self.target_distance)

        return total_dv + distance_error * 15.0  # Penalty weight

    def plan_two_impulse_maneuver(self, sat_A, sat_B, target_distance, transfer_time=3600.0,
                                  candidate_T_override=None, dv_bound=0.05):
        """Plans an optimal two-impulse maneuver with multi-start, bounds and fallback.
        candidate_T_override: Optional[List[float]] to override candidate transfer times (seconds)
        dv_bound: per-component absolute bound for ΔV (km/s)
        """
        print(f"\nPlanning maneuver for {sat_A.sat_id} to reach {target_distance}km from {sat_B.sat_id}...")
        self.target_distance = target_distance

        best = None
        best_val = np.inf

        # Candidate transfer times (s): include given plus some common windows, or override
        default_T = [transfer_time, 1200.0, 1800.0, 2400.0, 3600.0]
        candidate_T = sorted(set(candidate_T_override if candidate_T_override is not None else default_T))

        # Heuristic initial guess: first burn towards target line, small terminal tweak
        rel = sat_B.position - sat_A.position
        rel_dir = rel / np.linalg.norm(rel) if np.linalg.norm(rel) > 0 else np.array([1.0, 0.0, 0.0])
        base_guess = np.hstack([rel_dir * 0.005, np.zeros(3)])  # ~5 m/s magnitude

        # Bounds per component: ±dv_bound (km/s)
        bounds = [(-dv_bound, dv_bound)] * 6

        for T in candidate_T:
            # Multi-start (1 heuristic + 5 random perturbations)
            for k in range(6):
                if k == 0:
                    x0 = base_guess
                else:
                    x0 = base_guess + (np.random.rand(6) - 0.5) * 0.02  # ±10 m/s perturbation

                try:
                    res = minimize(
                        self._cost_function,
                        x0,
                        args=(sat_A.state_vector, sat_B.state_vector, T),
                        method='SLSQP',
                        bounds=bounds,
                        options={'disp': False, 'maxiter': 1000}
                    )
                except Exception:
                    res = None

                if res is not None and res.success and res.fun < best_val:
                    best = (res.x[:3], res.x[3:], T)
                    best_val = res.fun

            # Early stop if good enough at this T
            if np.isfinite(best_val) and best_val < 50.0:
                break

        # Fallback: Nelder-Mead (unconstrained) at original T
        if best is None:
            try:
                res2 = minimize(
                    self._cost_function,
                    base_guess,
                    args=(sat_A.state_vector, sat_B.state_vector, transfer_time),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
                )
                if res2.success:
                    best = (res2.x[:3], res2.x[3:], transfer_time)
                    best_val = res2.fun
            except Exception:
                pass

        if best is not None:
            dv1, dv2, Tsel = best
            print(f"Optimization successful!")
            print(f"  - Transfer time: {Tsel:.0f} s")
            print(f"  - Delta-V 1: {np.linalg.norm(dv1) * 1000:.2f} m/s")
            print(f"  - Delta-V 2: {np.linalg.norm(dv2) * 1000:.2f} m/s")
            print(f"  - Total Delta-V: {(np.linalg.norm(dv1) + np.linalg.norm(dv2)) * 1000:.2f} m/s")
            return dv1, dv2, Tsel

        print("Optimization failed!")
        return None, None, None

    def run(self):
        self.setup_initial_scene()

        # --- Step 1: Blue satellite maneuvers to 20km standoff ---
        # Randomly select a blue satellite to maneuver
        self.blue_maneuver_star = self.satellites[f'blue_{np.random.randint(1, 6)}']
        print(f"\n--- STEP 1: INITIATING MANEUVER ---")
        print(f"Selected {self.blue_maneuver_star.sat_id} to approach {self.red_main_star.sat_id}.")
        self._log_satellite_states("STEP1_START")

        # Plan the maneuver
        dv1, dv2, transfer_time = self.plan_two_impulse_maneuver(
            self.blue_maneuver_star,
            self.red_main_star,
            target_distance=20.0,
            transfer_time=3600.0  # Maneuver takes 1 hour
        )

        if dv1 is None:
            print("Maneuver planning failed. Aborting simulation.")
            return

        # Execute the maneuver over the simulation time
        # Apply first impulse
        self.blue_maneuver_star.state_vector[3:] += dv1
        self.blue_maneuver_star.is_maneuvering = True
        self._log(f"[T={self.time:.1f}s] {self.blue_maneuver_star.sat_id} 第一次脉冲 Δv={np.linalg.norm(dv1)*1000:.2f} m/s")
        self._log_satellite_states("STEP1_AFTER_DV1")

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
                print(
                    f"  [T_sim={self.time:.0f}s] Distance between {self.blue_maneuver_star.sat_id} and {self.red_main_star.sat_id}: {dist:.2f} km")
                self._log_satellite_states("STEP1_TICK")

        # Apply second impulse
        self.blue_maneuver_star.state_vector[3:] += dv2
        self.blue_maneuver_star.is_maneuvering = False
        self._log(f"[T={self.time:.1f}s] {self.blue_maneuver_star.sat_id} 第二次脉冲 Δv={np.linalg.norm(dv2)*1000:.2f} m/s")
        self._log_satellite_states("STEP1_AFTER_DV2")

        final_dist = np.linalg.norm(self.blue_maneuver_star.position - self.red_main_star.position)
        print(f"--- STEP 1: MANEUVER COMPLETE ---")
        print(f"Final distance to {self.red_main_star.sat_id}: {final_dist:.2f} km")
        # 若未达20km，进行多次小步终端微调并短传播以就位
        tweak_tries = 0
        while abs(final_dist - 20.0) > 0.3 and tweak_tries < 3:
            rel = self.blue_maneuver_star.position - self.red_main_star.position
            d = np.linalg.norm(rel)
            if d <= 1e-6:
                break
            dir_unit = rel / d
            # 误差方向：太远往内、太近往外（更温和的比例，叠代多次）
            dv_fix = np.clip((20.0 - d) * 0.0008, -0.02, 0.02)  # km/s，±20 m/s 上限
            self.blue_maneuver_star.state_vector[3:] += dir_unit * dv_fix
            fix_end = self.time + 400.0
            while self.time < fix_end:
                for sat in self.satellites.values():
                    sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
                self.time += self.dt
            final_dist = np.linalg.norm(self.blue_maneuver_star.position - self.red_main_star.position)
            print(f"After terminal tweak #{tweak_tries + 1} (blue), distance: {final_dist:.2f} km")
            self._log_satellite_states(f"STEP1_TWEAK_{tweak_tries+1}")
            tweak_tries += 1

        # --- Station-Keeping (悬停：保持20km相对距离) ---
        print(f"\n--- STATION-KEEPING: {self.blue_maneuver_star.sat_id} 在 20km 处悬停 ---")
        station_keeping_end = self.time + SimConfig.station_duration
        TARGET_DIST = SimConfig.station_target_dist
        TOL = SimConfig.station_tol_km
        # PD控制参数（单位配比成直接Δv增量，避免再乘dt）
        KP = SimConfig.station_kp
        KV = SimConfig.station_kv
        KI = SimConfig.station_ki
        DV_STEP_MAX = SimConfig.station_dv_step_max
        err_int = 0.0  # 误差积分（径向）
        err_int_max = SimConfig.station_err_int_max

        while self.time < station_keeping_end:
            # 先推进其他卫星
            for sat in self.satellites.values():
                if sat is self.blue_maneuver_star:
                    continue
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))

            # 推进蓝方机动星一步
            new_state = lagrange_extrapolate(self.blue_maneuver_star.state_vector, self.dt)
            self.blue_maneuver_star.update_state(new_state)

            # 计算与红方主星的距离与相对速度，并做PD校正
            rel = self.blue_maneuver_star.position - self.red_main_star.position
            dist = np.linalg.norm(rel)
            err = dist - TARGET_DIST
            if dist > 1e-6:
                dir_unit = rel / dist
                vel_rel = self.blue_maneuver_star.velocity - self.red_main_star.velocity
                v_radial = float(np.dot(vel_rel, dir_unit))
                # 死区与微分限幅
                if abs(err) < SimConfig.station_dead_err and abs(v_radial) < SimConfig.station_dead_vr:
                    dv_cmd = 0.0
                else:
                    v_radial_eff = np.clip(v_radial, -0.2, 0.2)
                    # 积分项（只在误差不太大时积累）
                    if abs(err) < 5.0:
                        err_int = np.clip(err_int + err * self.dt, -err_int_max, err_int_max)
                    # PI+D 控制（径向）
                    dv_cmd = (-KP * err - KI * err_int - KV * v_radial_eff)  # km/s
                # 限幅
                dv_cmd = max(-DV_STEP_MAX, min(DV_STEP_MAX, dv_cmd))
                self.blue_maneuver_star.state_vector[3:] += dir_unit * dv_cmd

            self.time += self.dt
            if int(self.time) % 300 == 0:
                print(f"  [T_sim={self.time:.0f}s] Hover distance: {dist:.2f} km (err={err:.2f} km)")
                self._log_satellite_states("HOVER_TICK")

        print("--- STATION-KEEPING COMPLETE ---")
        self._log_satellite_states("HOVER_DONE")

        # --- Step 2: Red protector satellite maneuvers to 5km to fly-around ---
        print(f"\n--- STEP 2: INITIATING INTERCEPT ---")
        print(f"{self.red_protector_star.sat_id} is moving to intercept {self.blue_maneuver_star.sat_id}.")
        self._log_satellite_states("STEP2_START")

        # Plan the intercept maneuver
        dv1_red, dv2_red, transfer_time_red = self.plan_two_impulse_maneuver(
            self.red_protector_star,
            self.blue_maneuver_star,
            target_distance=5.0,
            transfer_time=SimConfig.intercept_transfer_time  # Intercept takes 30 minutes
        )

        if dv1_red is None:
            print("Intercept maneuver planning failed. Aborting simulation.")
            return

        # Execute the intercept maneuver
        # Apply first impulse
        self.red_protector_star.state_vector[3:] += dv1_red
        self.red_protector_star.is_maneuvering = True
        self._log(f"[T={self.time:.1f}s] {self.red_protector_star.sat_id} 第一次脉冲 Δv={np.linalg.norm(dv1_red)*1000:.2f} m/s")
        self._log_satellite_states("STEP2_AFTER_DV1")

        print(f"\nExecuting intercept for {transfer_time_red} seconds...")
        intercept_end_time = self.time + transfer_time_red
        while self.time < intercept_end_time:
            # Propagate all satellites
            for sat in self.satellites.values():
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))

            self.time += self.dt
            dist = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
            if int(self.time) % 300 == 0:
                print(
                    f"  [T_sim={self.time:.0f}s] Distance between {self.red_protector_star.sat_id} and {self.blue_maneuver_star.sat_id}: {dist:.2f} km")
                self._log_satellite_states("STEP2_TICK")

        # Apply second impulse
        self.red_protector_star.state_vector[3:] += dv2_red
        self.red_protector_star.is_maneuvering = False
        self._log(f"[T={self.time:.1f}s] {self.red_protector_star.sat_id} 第二次脉冲 Δv={np.linalg.norm(dv2_red)*1000:.2f} m/s")
        self._log_satellite_states("STEP2_AFTER_DV2")

        final_dist_intercept = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
        print(f"--- STEP 2: INTERCEPT COMPLETE ---")
        print(f"Final distance to {self.blue_maneuver_star.sat_id}: {final_dist_intercept:.2f} km")

        # 若未达到5km，使用相对PD逼近（短时间闭环，避免盲脉冲）
        if final_dist_intercept > 5.5:
            print("Starting short relative PD converge to 5 km...")
            PD_duration = SimConfig.pd_duration
            end_pd = self.time + PD_duration
            kpr = SimConfig.pd_kpr
            kvv = SimConfig.pd_kv
            dv_max = SimConfig.pd_dv_max
            while self.time < end_pd:
                # 推进蓝星（保持悬停控制）
                # 先更新所有非红护卫星
                for sat in self.satellites.values():
                    if sat is self.red_protector_star:
                        continue
                    sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))

                # 红护卫星相对PD
                rel = self.blue_maneuver_star.position - self.red_protector_star.position
                d = np.linalg.norm(rel)
                if d < 1e-6:
                    break
                ex = rel / d
                v_rel = self.blue_maneuver_star.velocity - self.red_protector_star.velocity
                v_r = float(np.dot(v_rel, ex))
                # 符号修正：ex 从红指向蓝，d>5 时应沿 +ex 方向靠近
                dv_cmd = +kpr * (d - 5.0) - kvv * v_r
                dv_cmd = max(-dv_max, min(dv_max, dv_cmd))
                self.red_protector_star.state_vector[3:] += ex * dv_cmd

                # 推进红护卫星
                self.red_protector_star.update_state(
                    lagrange_extrapolate(self.red_protector_star.state_vector, self.dt))
                self.time += self.dt

                if int(self.time) % 300 == 0:
                    cur = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
                    print(f"  [T_sim={self.time:.0f}s] Converging distance: {cur:.2f} km")
                    self._log_satellite_states("STEP2_PD_TICK")

            final_dist_intercept = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
            print(f"After relative PD converge, distance: {final_dist_intercept:.2f} km")

        self.fly_around_start_time = self.time

        # --- Fly-Around Phase (护卫星绕飞蓝星) ---
        print(f"\n--- FLY-AROUND: red_protector 绕飞 {self.blue_maneuver_star.sat_id} ---")
        end_fly = self.time + SimConfig.fly_time
        # 期望角速度（绕飞一圈）
        omega = 2 * np.pi / (SimConfig.fly_period_hours * 3600.0)
        # 径向更强、切向更温和
        kpr_f = SimConfig.fly_kpr
        kv_r_f = SimConfig.fly_kv_r
        kpt_f = SimConfig.fly_kpt
        dv_max_f = SimConfig.fly_dv_max
        dv_prev = np.zeros(3)
        z_axis = np.array([0.0, 0.0, 1.0])
        while self.time < end_fly:
            # 推进蓝星（保持悬停规律）
            for sat in self.satellites.values():
                if sat is self.red_protector_star:
                    continue
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))

            # 计算相对基变量
            rel = self.red_protector_star.position - self.blue_maneuver_star.position
            d = np.linalg.norm(rel)
            if d < 1e-6:
                break
            ex = rel / d
            # 构造切向方向：优先采用相对速度在法向平面投影，避免奇异
            v_rel = self.red_protector_star.velocity - self.blue_maneuver_star.velocity
            v_rel_t = v_rel - np.dot(v_rel, ex) * ex
            if np.linalg.norm(v_rel_t) > 1e-6:
                et = v_rel_t / np.linalg.norm(v_rel_t)
            else:
                et = np.cross(z_axis, ex)
                if np.linalg.norm(et) < 1e-6:
                    et = np.cross(np.array([1.0, 0.0, 0.0]), ex)
                et = et / np.linalg.norm(et)
            v_r = float(np.dot(v_rel, ex))
            v_t = float(np.dot(v_rel, et))

            # 径向PD使 d -> 5 km；切向跟踪目标速度 v_tgt = omega * d
            dv_r = -kpr_f * (d - 5.0) - kv_r_f * v_r
            v_tgt = omega * d
            dv_t = kpt_f * (v_tgt - v_t)

            # 仅当半径误差较小才分配切向分量
            dv_r_limited = np.clip(dv_r, -dv_max_f, dv_max_f)
            if abs(d - 5.0) < SimConfig.fly_gate_err:
                rem = max(0.0, dv_max_f - abs(dv_r_limited))
                dv_t_limited = np.clip(dv_t, -rem, rem)
            else:
                dv_t_limited = 0.0
            dv_new = ex * dv_r_limited + et * dv_t_limited
            # 一阶低通抑制抖动（更强平滑）
            dv_vec = 0.85 * dv_prev + 0.15 * dv_new
            dv_prev = dv_vec

            self.red_protector_star.state_vector[3:] += dv_vec
            self.red_protector_star.update_state(lagrange_extrapolate(self.red_protector_star.state_vector, self.dt))

            self.time += self.dt
            if int(self.time) % 300 == 0:
                print(f"  [T_sim={self.time:.0f}s] Fly-around distance: {d:.2f} km, v_t={v_t:.3f} km/s")
                self._log_satellite_states("FLY_TICK")

        print("\nSIMULATION FINISHED.")
        self._log_satellite_states("FINISHED")
        try:
            self.log_f.flush()
            self.log_f.close()
        except Exception:
            pass

    def _execute_intercept_and_escape(self):
        # --- Step 2: Red protector satellite maneuvers to 5km to fly-around ---
        print(f"\n--- STEP 2: INITIATING INTERCEPT ---")
        print(f"{self.red_protector_star.sat_id} is moving to intercept {self.blue_maneuver_star.sat_id}.")

        # Plan the intercept maneuver
        dv1_red, dv2_red, transfer_time_red = self.plan_two_impulse_maneuver(
            self.red_protector_star,
            self.blue_maneuver_star,
            target_distance=5.0,
            transfer_time=1800.0  # Intercept takes 30 minutes
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
        fly_around_duration = 900.0  # 15 minutes
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
            tangential_dir = np.cross(radial_dir, [0, 0, 1])  # Simplified tangential direction
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
        EVASION_DV_STEP = 0.001  # km/s per step

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

