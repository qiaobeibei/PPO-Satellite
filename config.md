

参数清单（SimConfig）
- 基础
  - dt: 仿真步长（s）
  - log_file: 日志文件名
- 悬停（蓝星在红主星20 km）
  - station_target_dist: 目标距离（km）
  - station_duration: 悬停时长（s）
  - station_tol_km: 仅用于显示的误差带（km）
  - station_kp/kv/ki: 径向 PI+D 增益
  - station_dv_step_max: 每步最大Δv（km/s）
  - station_dead_err: 死区半径误差阈值（km）
  - station_dead_vr: 死区径向速度阈值（km/s）
  - station_err_int_max: 积分抗饱和上限（km*s）
- 拦截（红护卫→蓝机动星5 km）
  - intercept_transfer_time: 两脉冲初始规划的转移时间（s）
  - pd_duration: PD收敛时长（s）
  - pd_kpr/pd_kv: 径向 PD 增益
  - pd_dv_max: PD每步最大Δv（km/s）
- 绕飞（5 km 半径相对环绕）
  - fly_time: 绕飞仿真时长（s）
  - fly_period_hours: 目标绕飞周期（小时），越大越稳
  - fly_kpr/fly_kv_r: 径向 PD 增益
  - fly_kpt: 切向 P 增益（跟踪 v_t=ω·d）
  - fly_dv_max: 绕飞每步最大Δv（km/s）
  - fly_gate_err: 只有当 |d-5| < fly_gate_err 时分配切向分量

如何调参（简明建议）
- 悬停距离抖动大或偏离慢
  - 增大 station_kp 可更快贴近；增大 station_kv 抑制振荡。
  - 如果存在稳态偏差（总偏外或偏内），稍增 station_ki；若有积分累积过大，降低 station_ki 或增大 station_err_int_max 并减小 station_dv_step_max。
  - 悬停过“激进”可减小 station_dv_step_max。
- 拦截末距>5 km 难以收紧
  - 先调 intercept_transfer_time（或在你需要时把 planner 中候选 T 扩大），可提升两脉冲的接近度。
  - PD阶段慢：增大 pd_kpr；抖动/过冲：增大 pd_kv、减小 pd_dv_max。
- 绕飞半径发散或大幅波动
  - 降低 fly_kpt（切向）；提高 fly_kpr/fly_kv_r（径向优先）；减小 fly_dv_max（更稳但收敛慢）。
  - 增大 fly_period_hours（角速度更小）也会更稳。
  - 收敛门限更严格：减小 fly_gate_err，使切向只在更近时启用。
- 整体数值不稳定
  - 减小 dt（更高的数值稳定性，代价是耗时）；同时调小各类 dv_max。
