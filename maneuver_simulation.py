"""
Author: beauqiao
Created: 2025-09-08
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
from satellite_function import Time_window_of_danger_zone



class SimConfig:
    dt = 10.0 # 时间步长 (s)
    print_tick_sec = 600  # 控制台打印与日志tick间隔（秒）
    plot_sample_sec = 60   # 轨迹采样间隔（秒）
    log_file = 'log/maneuver_log.txt'
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
    # 初始相对距离
    initial_blue_relative_distance_km = 50.0
    initial_red_relative_distance_km = 2.0

    # 编队规模与任务分配
    total_blue_count = 5
    total_red_protector_count = 2
    num_blue_to_approach = 5
    num_red_to_flyaround = 2
    # 顺序/并发机动开关
    concurrent_mode = True

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
        os.makedirs(os.path.dirname(SimConfig.log_file), exist_ok=True)
        os.makedirs('result', exist_ok=True)
        self.log_f = open(SimConfig.log_file, 'w', encoding='utf-8')
        # 采样与绘图相关
        self.selected_blue_ids = []
        self.assigned_red_ids = set()
        self.trace_t = []
        self.trace_pos = {}  # sat_id -> list of np.array([x,y,z])

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

    def plog(self, text):
        print(text)
        self._log(text)

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
        r, v = self.red_main_star.position, self.red_main_star.velocity
        r_norm = r / np.linalg.norm(r)
        v_norm = v / np.linalg.norm(v)
        c_norm = np.cross(r_norm, v_norm)

        # 红方护卫星
        self.red_protectors = []
        for k in range(SimConfig.total_red_protector_count):
            # 交替分布在 ±v、±c、+r 等方向，避免完全重合
            dirs = [v_norm, -v_norm, c_norm, -c_norm, r_norm]
            d = dirs[k % len(dirs)]
            pos = r + d * SimConfig.initial_red_relative_distance_km
            vel = v
            prot = Satellite(f'red_protector_{k+1}', 'red', np.hstack([pos, vel]))
            self.satellites[prot.sat_id] = prot
            self.red_protectors.append(prot)

        # 蓝方5颗卫星
        offsets = [
            r_norm * SimConfig.initial_blue_relative_distance_km,  # 上方
            -r_norm * SimConfig.initial_blue_relative_distance_km,  # 下方
            c_norm * SimConfig.initial_blue_relative_distance_km,  # 左侧
            -c_norm * SimConfig.initial_blue_relative_distance_km,  # 右侧
            v_norm * SimConfig.initial_blue_relative_distance_km  # 前方
        ]
        for i in range(SimConfig.total_blue_count):
            blue_pos = r + offsets[i]
            blue_vel = v  # 简化处理，速度与主星相同
            blue_sat = Satellite(f'blue_{i + 1}', 'blue', np.hstack([blue_pos, blue_vel]))
            self.satellites[f'blue_{i + 1}'] = blue_sat

        self.plog("场景初始化完成，卫星部署如下：")
        for name, sat in self.satellites.items():
            dist = np.linalg.norm(sat.position - self.red_main_star.position)
            self.plog(f"- {name}: 距离主星 {dist:.2f} km")
        # 记录初始化
        self._log_satellite_states("INIT")
        # 采样初始帧（此时仅主星记录）
        self._trace_sample()

    def _trace_sample(self):
        """按采样间隔记录主星与相关机动卫星的位置轨迹"""
        if int(self.time) % SimConfig.plot_sample_sec != 0:
            return
        self.trace_t.append(self.time)
        ids = set()
        if self.red_main_star is not None:
            ids.add(self.red_main_star.sat_id)
        ids.update(self.selected_blue_ids)
        ids.update(self.assigned_red_ids)
        # 确保所有已知ID长度对齐（缺失填充NaN）
        all_ids = set(self.trace_pos.keys()) | ids
        for sid in all_ids:
            if sid not in self.trace_pos:
                self.trace_pos[sid] = []
            # 补齐至当前时间索引-1
            while len(self.trace_pos[sid]) < len(self.trace_t) - 1:
                self.trace_pos[sid].append(np.array([np.nan, np.nan, np.nan]))
            sat = self.satellites.get(sid)
            if sat is not None and sid in ids:
                self.trace_pos[sid].append(sat.position.copy())
            else:
                self.trace_pos[sid].append(np.array([np.nan, np.nan, np.nan]))

    def _plot_results(self):
        if not self.trace_t:
            return
        # 3D 轨迹图
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        # 主星
        rm_id = self.red_main_star.sat_id if self.red_main_star is not None else None
        if rm_id in self.trace_pos:
            r = np.array(self.trace_pos[rm_id])
            ax.plot(r[:,0], r[:,1], r[:,2], color='r', linestyle='--', linewidth=1.5, label='red_main')
            # 起点标注
            ax.scatter(r[0,0], r[0,1], r[0,2], color='r', s=20)
            ax.text(r[0,0], r[0,1], r[0,2], 'start', fontsize=8, color='r')
        # 蓝机动集群
        for sid in self.selected_blue_ids:
            if sid in self.trace_pos:
                r = np.array(self.trace_pos[sid])
                ax.plot(r[:,0], r[:,1], r[:,2], color='b', linewidth=1.0, label=sid)
                ax.scatter(r[0,0], r[0,1], r[0,2], color='b', s=14)
                ax.text(r[0,0], r[0,1], r[0,2], f'{sid}_start', fontsize=8, color='b')
        # 红护卫机动
        for sid in self.assigned_red_ids:
            if sid in self.trace_pos:
                r = np.array(self.trace_pos[sid])
                ax.plot(r[:,0], r[:,1], r[:,2], color='r', linewidth=1.0, label=sid)
                ax.scatter(r[0,0], r[0,1], r[0,2], color='r', s=14)
                ax.text(r[0,0], r[0,1], r[0,2], f'{sid}_start', fontsize=8, color='r')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join('result', 'traj_3d.png'), dpi=150)
        plt.close(fig)

        # 距离-时间图：蓝星-主星
        if rm_id in self.trace_pos:
            rm = np.array(self.trace_pos[rm_id])
            fig2 = plt.figure(figsize=(8,4))
            for sid in self.selected_blue_ids:
                if sid not in self.trace_pos:
                    continue
                b = np.array(self.trace_pos[sid])
                n = min(len(b), len(rm), len(self.trace_t))
                b = b[:n]
                rm_n = rm[:n]
                t = np.array(self.trace_t[:n])
                mask = (~np.isnan(b).any(axis=1)) & (~np.isnan(rm_n).any(axis=1))
                if not np.any(mask):
                    continue
                d = np.linalg.norm(b[mask] - rm_n[mask], axis=1)
                plt.plot(t[mask]/3600.0, d, label=f'{sid}-red_main')
            plt.xlabel('Time (hour)')
            plt.ylabel('Distance (km)')
            plt.title('Blue-to-RedMain Distance')
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join('result', 'dist_blue_redmain.png'), dpi=150)
            plt.close(fig2)

    def intercept_and_flyaround_current_pair(self):
        # 规划拦截
        dv1_red, dv2_red, transfer_time_red = self.plan_two_impulse_maneuver(
            self.red_protector_star,
            self.blue_maneuver_star,
            target_distance=5.0,
            transfer_time=SimConfig.intercept_transfer_time
        )

        if dv1_red is None:
            print("Intercept maneuver planning failed. Skip this protector.")
            return

        # 执行拦截：第一次脉冲
        self.red_protector_star.state_vector[3:] += dv1_red
        self.red_protector_star.is_maneuvering = True
        self._log(f"[T={self.time:.1f}s][STEP2] {self.red_protector_star.sat_id} 第一次脉冲 Δv={np.linalg.norm(dv1_red)*1000:.2f} m/s")
        self._log_satellite_states("STEP2_AFTER_DV1")

        print(f"\nExecuting intercept for {transfer_time_red} seconds...")
        intercept_end_time = self.time + transfer_time_red
        while self.time < intercept_end_time:
            for sat in self.satellites.values():
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
            self.time += self.dt
            if int(self.time) % SimConfig.print_tick_sec == 0:
                dcur = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
                print(f"  [T_sim={self.time:.0f}s] Distance between {self.red_protector_star.sat_id} and {self.blue_maneuver_star.sat_id}: {dcur:.2f} km")
                self._log_satellite_states("STEP2_TICK")

        # 第二次脉冲
        self.red_protector_star.state_vector[3:] += dv2_red
        self.red_protector_star.is_maneuvering = False
        self._log(f"[T={self.time:.1f}s][STEP2] {self.red_protector_star.sat_id} 第二次脉冲 Δv={np.linalg.norm(dv2_red)*1000:.2f} m/s")
        self._log_satellite_states("STEP2_AFTER_DV2")

        final_dist_intercept = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
        print(f"--- STEP 2: INTERCEPT COMPLETE ---")
        print(f"Final distance to {self.blue_maneuver_star.sat_id}: {final_dist_intercept:.2f} km")

        # 若未达到5km，使用相对PD逼近
        if final_dist_intercept > 5.5:
            print("Starting short relative PD converge to 5 km...")
            PD_duration = SimConfig.pd_duration
            end_pd = self.time + PD_duration
            kpr = SimConfig.pd_kpr
            kvv = SimConfig.pd_kv
            dv_max = SimConfig.pd_dv_max
            while self.time < end_pd:
                for sat in self.satellites.values():
                    if sat is self.red_protector_star:
                        continue
                    sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))

                rel = self.blue_maneuver_star.position - self.red_protector_star.position
                d = np.linalg.norm(rel)
                if d < 1e-6:
                    break
                ex = rel / d
                v_rel = self.blue_maneuver_star.velocity - self.red_protector_star.velocity
                v_r = float(np.dot(v_rel, ex))
                dv_cmd = +kpr * (d - 5.0) - kvv * v_r
                dv_cmd = max(-dv_max, min(dv_max, dv_cmd))
                self.red_protector_star.state_vector[3:] += ex * dv_cmd

                self.red_protector_star.update_state(lagrange_extrapolate(self.red_protector_star.state_vector, self.dt))
                self.time += self.dt

                if int(self.time) % SimConfig.print_tick_sec == 0:
                    cur = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
                    print(f"  [T_sim={self.time:.0f}s] Converging distance: {cur:.2f} km")
                    self._log_satellite_states("STEP2_PD_TICK")

            final_dist_intercept = np.linalg.norm(self.red_protector_star.position - self.blue_maneuver_star.position)
            print(f"After relative PD converge, distance: {final_dist_intercept:.2f} km")

        self.fly_around_start_time = self.time

        # 绕飞阶段（与主流程一致，使用配置参数）
        print(f"\n--- FLY-AROUND: {self.red_protector_star.sat_id} 绕飞 {self.blue_maneuver_star.sat_id} ---")
        end_fly = self.time + SimConfig.fly_time
        omega = 2 * np.pi / (SimConfig.fly_period_hours * 3600.0)
        kpr_f = SimConfig.fly_kpr
        kv_r_f = SimConfig.fly_kv_r
        kpt_f = SimConfig.fly_kpt
        dv_max_f = SimConfig.fly_dv_max
        dv_prev = np.zeros(3)
        z_axis = np.array([0.0, 0.0, 1.0])
        while self.time < end_fly:
            for sat in self.satellites.values():
                if sat is self.red_protector_star:
                    continue
                sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))

            rel = self.red_protector_star.position - self.blue_maneuver_star.position
            d = np.linalg.norm(rel)
            if d < 1e-6:
                break
            ex = rel / d
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

            dv_r = -kpr_f * (d - 5.0) - kv_r_f * v_r
            v_tgt = omega * d
            dv_t = kpt_f * (v_tgt - v_t)

            dv_r_limited = np.clip(dv_r, -dv_max_f, dv_max_f)
            if abs(d - 5.0) < SimConfig.fly_gate_err:
                rem = max(0.0, dv_max_f - abs(dv_r_limited))
                dv_t_limited = np.clip(dv_t, -rem, rem)
            else:
                dv_t_limited = 0.0
            dv_new = ex * dv_r_limited + et * dv_t_limited
            dv_vec = 0.85 * dv_prev + 0.15 * dv_new
            dv_prev = dv_vec

            self.red_protector_star.state_vector[3:] += dv_vec
            self.red_protector_star.update_state(lagrange_extrapolate(self.red_protector_star.state_vector, self.dt))

            self.time += self.dt
            if int(self.time) % SimConfig.print_tick_sec == 0:
                print(f"  [T_sim={self.time:.0f}s] Fly-around distance: {d:.2f} km, v_t={v_t:.3f} km/s")
                self._log_satellite_states("FLY_TICK")

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

        # --- 任务分配 ---
        # 选出若干蓝星作为抵近对象
        blue_ids = [f'blue_{i+1}' for i in range(SimConfig.total_blue_count)]
        np.random.shuffle(blue_ids)
        selected_blues = blue_ids[:SimConfig.num_blue_to_approach]
        self.selected_blue_ids = selected_blues[:]
        # 为每个蓝目标分配若干红护卫
        red_list = self.red_protectors.copy()
        np.random.shuffle(red_list)

        self.plog(f"\n--- TASK ASSIGNMENT ---")
        self.plog(f"Blue to approach (count={len(selected_blues)}): {selected_blues}")

        if SimConfig.concurrent_mode:
            # --- STEP 1 并发：为所有被选蓝星同时规划与执行 ---
            blue_objs = [self.satellites[bid] for bid in selected_blues]
            self.plog("\n--- STEP 1: INITIATING MANEUVER (CONCURRENT) ---")
            for b in blue_objs:
                self.plog(f"Selected {b.sat_id} to approach {self.red_main_star.sat_id}.")
            # 规划
            blue_plans = {}
            for b in blue_objs:
                dv1, dv2, T = self.plan_two_impulse_maneuver(b, self.red_main_star, target_distance=20.0, transfer_time=3600.0)
                if dv1 is None:
                    self.plog(f"{b.sat_id} maneuver planning failed. Abort.")
                    return
                blue_plans[b.sat_id] = {"dv1": dv1, "dv2": dv2, "T": T, "dv2_applied": False, "t_end": self.time + T}
            # 一次性施加第一次脉冲
            for b in blue_objs:
                p = blue_plans[b.sat_id]
                b.state_vector[3:] += p["dv1"]
                b.is_maneuvering = True
            self._log_satellite_states("STEP1_AFTER_ALL_DV1")
            # 采样首帧
            self._trace_sample()
            # 推进，达到各自结束时刻施加第二次脉冲
            while True:
                all_done = True
                for b in self.satellites.values():
                    b.update_state(lagrange_extrapolate(b.state_vector, self.dt))
                self.time += self.dt
                self._trace_sample()
                for b in blue_objs:
                    p = blue_plans[b.sat_id]
                    if not p["dv2_applied"] and self.time >= p["t_end"]:
                        b.state_vector[3:] += p["dv2"]
                        b.is_maneuvering = False
                        p["dv2_applied"] = True
                    if not p["dv2_applied"]:
                        all_done = False
                if int(self.time) % SimConfig.print_tick_sec == 0:
                    for b in blue_objs:
                        dist = np.linalg.norm(b.position - self.red_main_star.position)
                        print(f"  [T_sim={self.time:.0f}s][{b.sat_id}] Distance to red_main: {dist:.2f} km")
                    self._log_satellite_states("STEP1_TICK")
                if all_done:
                    break
            self._log_satellite_states("STEP1_AFTER_ALL_DV2")

            # 并行悬停
            self.plog("\n--- STATION-KEEPING (CONCURRENT) ---")
            station_end = self.time + SimConfig.station_duration
            hover_err_int = {b.sat_id: 0.0 for b in blue_objs}
            z_axis = np.array([0.0, 0.0, 1.0])
            while self.time < station_end:
                for name, sat in self.satellites.items():
                    if sat in blue_objs:
                        continue
                    sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
                for b in blue_objs:
                    new_state = lagrange_extrapolate(b.state_vector, self.dt)
                    b.update_state(new_state)
                    rel = b.position - self.red_main_star.position
                    dist = np.linalg.norm(rel)
                    err = dist - SimConfig.station_target_dist
                    if dist > 1e-6:
                        ex = rel / dist
                        vel_rel = b.velocity - self.red_main_star.velocity
                        v_r = float(np.dot(vel_rel, ex))
                        v_t_vec = vel_rel - v_r * ex
                        if np.linalg.norm(v_t_vec) > 1e-6:
                            et = v_t_vec / np.linalg.norm(v_t_vec)
                        else:
                            et = np.cross(z_axis, ex)
                            if np.linalg.norm(et) < 1e-6:
                                et = np.cross(np.array([1.0, 0.0, 0.0]), ex)
                            et = et / np.linalg.norm(et)
                        if abs(err) < SimConfig.station_dead_err and abs(v_r) < SimConfig.station_dead_vr:
                            dv_r = 0.0
                        else:
                            v_r_eff = np.clip(v_r, -0.2, 0.2)
                            if abs(err) < 5.0:
                                hover_err_int[b.sat_id] = np.clip(hover_err_int[b.sat_id] + err * self.dt, -SimConfig.station_err_int_max, SimConfig.station_err_int_max)
                            dv_r = (-SimConfig.station_kp * err - SimConfig.station_ki * hover_err_int[b.sat_id] - SimConfig.station_kv * v_r_eff)
                        dv_r = np.clip(dv_r, -SimConfig.station_dv_step_max, SimConfig.station_dv_step_max)
                        dv_t = 0.0
                        # 可选切向控制：此处保持关闭，聚焦径向稳定
                        b.state_vector[3:] += ex * dv_r + et * dv_t
                self.time += self.dt
                self._trace_sample()
                if int(self.time) % SimConfig.print_tick_sec == 0:
                    for b in blue_objs:
                        dist = np.linalg.norm(b.position - self.red_main_star.position)
                        err = dist - SimConfig.station_target_dist
                        print(f"  [T_sim={self.time:.0f}s][{b.sat_id}] Hover distance: {dist:.2f} km (err={err:.2f} km)")
                    self._log_satellite_states("HOVER_TICK_ALL")
            self._log_satellite_states("HOVER_DONE_ALL")

            # --- STEP 2 并发：为每个蓝星分配护卫，并同时拦截/绕飞 ---
            assignments = []
            for b in blue_objs:
                num_assign = min(SimConfig.num_red_to_flyaround, len(red_list))
                assigned_reds = red_list[:num_assign]
                red_list = red_list[num_assign:]
                for rp in assigned_reds:
                    assignments.append((rp, b))
            # 记录将参与机动的红护卫ID集合
            self.assigned_red_ids = {rp.sat_id for rp, _ in assignments}
            if not assignments:
                self.plog("No available red protectors to assign.")
            else:
                self.plog("\n--- STEP 2: INITIATING INTERCEPT (CONCURRENT) ---")
                # 规划所有拦截
                plans = {}
                for rp, b in assignments:
                    self.plog(f"{rp.sat_id} is moving to intercept {b.sat_id}.")
                    dv1, dv2, T = self.plan_two_impulse_maneuver(rp, b, target_distance=5.0, transfer_time=SimConfig.intercept_transfer_time)
                    if dv1 is None:
                        self.plog(f"{rp.sat_id} intercept planning failed.")
                        continue
                    plans[rp.sat_id] = {"target": b.sat_id, "dv1": dv1, "dv2": dv2, "t_end": self.time + T, "dv2_applied": False}
                # 施加第一次脉冲
                for rp, b in assignments:
                    if rp.sat_id in plans:
                        rp.state_vector[3:] += plans[rp.sat_id]["dv1"]
                        rp.is_maneuvering = True
                self._log_satellite_states("STEP2_AFTER_ALL_DV1")
                # 推进，按各自结束时间施加第二次脉冲
                while True:
                    all_done = True
                    for sat in self.satellites.values():
                        sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
                    self.time += self.dt
                    self._trace_sample()
                    for rp, b in assignments:
                        if rp.sat_id not in plans:
                            continue
                        p = plans[rp.sat_id]
                        if not p["dv2_applied"] and self.time >= p["t_end"]:
                            rp.state_vector[3:] += p["dv2"]
                            rp.is_maneuvering = False
                            p["dv2_applied"] = True
                        if not p["dv2_applied"]:
                            all_done = False
                    if int(self.time) % SimConfig.print_tick_sec == 0:
                        for rp, b in assignments:
                            dcur = np.linalg.norm(rp.position - b.position)
                            print(f"  [T_sim={self.time:.0f}s][{rp.sat_id}->{b.sat_id}] Distance: {dcur:.2f} km")
                        self._log_satellite_states("STEP2_TICK")
                    if all_done:
                        break
                self._log_satellite_states("STEP2_AFTER_ALL_DV2")
                # 输出最终距离
                for rp, b in assignments:
                    dfin = np.linalg.norm(rp.position - b.position)
                    self.plog(f"--- STEP 2: INTERCEPT COMPLETE ---")
                    self.plog(f"Final distance {rp.sat_id}->{b.sat_id}: {dfin:.2f} km")
                # 并行绕飞（简化为短时并发控制）
                fly_end = self.time + min(SimConfig.fly_time, 600.0)
                dv_prev = {rp.sat_id: np.zeros(3) for rp, _ in assignments}
                z_axis = np.array([0.0, 0.0, 1.0])
                omega = 2 * np.pi / (SimConfig.fly_period_hours * 3600.0)
                while self.time < fly_end:
                    # 推进未机动对象
                    for sat in self.satellites.values():
                        sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
                    # 控制每个护卫
                    for rp, b in assignments:
                        rel = rp.position - b.position
                        d = np.linalg.norm(rel)
                        if d < 1e-6:
                            continue
                        ex = rel / d
                        v_rel = rp.velocity - b.velocity
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
                        dv_r = -SimConfig.fly_kpr * (d - 5.0) - SimConfig.fly_kv_r * v_r
                        v_tgt = omega * d
                        dv_t = SimConfig.fly_kpt * (v_tgt - v_t)
                        dv_r_limited = np.clip(dv_r, -SimConfig.fly_dv_max, SimConfig.fly_dv_max)
                        if abs(d - 5.0) < SimConfig.fly_gate_err:
                            rem = max(0.0, SimConfig.fly_dv_max - abs(dv_r_limited))
                            dv_t_limited = np.clip(dv_t, -rem, rem)
                        else:
                            dv_t_limited = 0.0
                        dv_new = ex * dv_r_limited + et * dv_t_limited
                        dv_vec = 0.85 * dv_prev[rp.sat_id] + 0.15 * dv_new
                        dv_prev[rp.sat_id] = dv_vec
                        rp.state_vector[3:] += dv_vec
                    self.time += self.dt
                    self._trace_sample()
                    if int(self.time) % SimConfig.print_tick_sec == 0:
                        for rp, b in assignments:
                            d = np.linalg.norm(rp.position - b.position)
                            print(f"  [T_sim={self.time:.0f}s][{rp.sat_id}->{b.sat_id}] Fly-around distance: {d:.2f} km")
                        self._log_satellite_states("FLY_TICK_ALL")
        else:
            # 回退：顺序执行（保持已有顺序版逻辑）
            for idx_b, blue_id in enumerate(selected_blues):
                self.blue_maneuver_star = self.satellites[blue_id]
                self.plog(f"\n--- STEP 1: INITIATING MANEUVER ---")
                self.plog(f"Selected {self.blue_maneuver_star.sat_id} to approach {self.red_main_star.sat_id}.")
                self._log_satellite_states("STEP1_START")
                dv1, dv2, transfer_time = self.plan_two_impulse_maneuver(
                    self.blue_maneuver_star,
                    self.red_main_star,
                    target_distance=20.0,
                    transfer_time=3600.0
                )
                if dv1 is None:
                    print("Maneuver planning failed. Aborting simulation.")
                    return
                self.blue_maneuver_star.state_vector[3:] += dv1
                self.blue_maneuver_star.is_maneuvering = True
                self._log(f"[T={self.time:.1f}s] {self.blue_maneuver_star.sat_id} 第一次脉冲 Δv={np.linalg.norm(dv1)*1000:.2f} m/s")
                self._log_satellite_states("STEP1_AFTER_DV1")
                maneuver_end_time = self.time + transfer_time
                while self.time < maneuver_end_time:
                    for sat in self.satellites.values():
                        sat.update_state(lagrange_extrapolate(sat.state_vector, self.dt))
                    self.time += self.dt
                self.blue_maneuver_star.state_vector[3:] += dv2
                self.blue_maneuver_star.is_maneuvering = False
                self._log(f"[T={self.time:.1f}s] {self.blue_maneuver_star.sat_id} 第二次脉冲 Δv={np.linalg.norm(dv2)*1000:.2f} m/s")
                self._log_satellite_states("STEP1_AFTER_DV2")

        self.plog("\nSIMULATION FINISHED.")
        self._log_satellite_states("FINISHED")
        # 绘图输出
        try:
            self._plot_results()
        except Exception as e:
            self.plog(f"Plotting failed: {e}")
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
            if int(self.time) % SimConfig.print_tick_sec == 0:
                print(f"  [T_sim={self.time:.0f}s] Pursuit Distance: {dist:.2f} km")

        final_dist_pursuit = np.linalg.norm(self.blue_maneuver_star.position - self.red_protector_star.position)
        print(f"--- PURSUIT-EVASION COMPLETE ---")
        print(f"Final distance between pursuer and evader: {final_dist_pursuit:.2f} km")
        print("\nSIMULATION FINISHED.")


if __name__ == '__main__':
    simulation = ManeuverSimulation()
    simulation.run()

