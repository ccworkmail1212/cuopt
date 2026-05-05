"""
Lot Scheduling Demo — 展示 NVIDIA cuOpt scheduling branch 全部 3 個新功能

問題背景：半導體晶圓廠，2 台機器（machines），4 個工單（lots）

工單：
  lot_0: 高優先（weight=4），截止 t=6 前必須開始
  lot_1: 中優先（weight=2），無截止限制
  lot_2: 中優先（weight=2），截止 t=8 前必須開始
  lot_3: 低優先（weight=1），無截止限制

機台偏好（set_vehicle_order_cost）：
  machine_0 擅長 lot_0/1（成本 0），不擅長 lot_2/3（成本 5）
  machine_1 擅長 lot_2/3（成本 0），不擅長 lot_0/1（成本 5）

目標：最小化加權完工時間（WCT）+ 截止違反懲罰 + 機台分配成本
"""

import numpy as np
import cudf
from cuopt import routing

print("=" * 60)
print("Lot Scheduling Demo（scheduling branch 全功能展示）")
print("=" * 60)

# 6 個 location：0=depot, 1-4=4個工單位置, 5=depot2(不用)
n_locations = 5   # depot + 4 lots
n_vehicles  = 2   # 2 台機器
n_orders    = 4   # 4 個工單

# Transit time（機台切換工單的準備時間）
transit = np.array([
    # dep  l0  l1  l2  l3
    [  0,   1,  1,  2,  2 ],  # depot
    [  1,   0,  1,  2,  3 ],  # lot_0
    [  1,   1,  0,  1,  2 ],  # lot_1
    [  2,   2,  1,  0,  1 ],  # lot_2
    [  2,   3,  2,  1,  0 ],  # lot_3
], dtype="float32")

cost = np.ones((n_locations, n_locations), dtype="float32")
np.fill_diagonal(cost, 0)

d = routing.DataModel(n_locations, n_vehicles, n_orders)
d.add_cost_matrix(cudf.DataFrame(cost))
d.add_transit_time_matrix(cudf.DataFrame(transit))

# 工單位置
d.set_order_locations(cudf.Series([1, 2, 3, 4], dtype="int32"))

# 加工時間
service_times = [3, 2, 4, 2]   # lot_0:3, lot_1:2, lot_2:4, lot_3:2
d.set_order_service_times(cudf.Series(service_times, dtype="int32"))

# 機台起終點（都從 depot 出發）
d.set_vehicle_locations(
    cudf.Series([0, 0], dtype="int32"),
    cudf.Series([0, 0], dtype="int32"),
)

# ── 功能 1：set_order_weights（加權完工時間）──────────────────
weights = [4, 2, 2, 1]
d.set_order_weights(cudf.Series(weights, dtype="int32"))
print()
print("功能 1：set_order_weights")
print(f"  lot_0: weight={weights[0]} (最高優先)")
print(f"  lot_1: weight={weights[1]}")
print(f"  lot_2: weight={weights[2]}")
print(f"  lot_3: weight={weights[3]} (最低優先)")

# ── 功能 2：set_order_due_times（截止時間）───────────────────
INF = 2147483647
due_times = [6, INF, 8, INF]   # lot_0 截止 t=6, lot_2 截止 t=8
d.set_order_due_times(cudf.Series(due_times, dtype="int32"))
print()
print("功能 2：set_order_due_times")
print(f"  lot_0: 必須在 t=6 前開始（否則有懲罰）")
print(f"  lot_1: 無截止限制")
print(f"  lot_2: 必須在 t=8 前開始（否則有懲罰）")
print(f"  lot_3: 無截止限制")

# ── 功能 3：set_vehicle_order_cost（機台偏好成本）───────────
# machine_0 擅長 lot_0/1（成本 0）；不擅長 lot_2/3（成本 5）
# machine_1 擅長 lot_2/3（成本 0）；不擅長 lot_0/1（成本 5）
m0_costs = cudf.Series([0, 0, 5, 5], dtype="int32")
m1_costs = cudf.Series([5, 5, 0, 0], dtype="int32")
d.set_vehicle_order_cost(0, m0_costs)
d.set_vehicle_order_cost(1, m1_costs)
print()
print("功能 3：set_vehicle_order_cost")
print(f"  machine_0 偏好 lot_0/1（成本 0），不擅長 lot_2/3（成本 5）")
print(f"  machine_1 偏好 lot_2/3（成本 0），不擅長 lot_0/1（成本 5）")

# ── 求解 ─────────────────────────────────────────────────────
print()
print("求解中...")
solver_settings = routing.SolverSettings()
solver_settings.set_time_limit(10.0)
sol = routing.Solve(d, solver_settings)

# ── 結果 ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("求解結果")
print("=" * 60)
status_map = {0: "Optimal", 1: "Feasible", 2: "Infeasible", 3: "Error"}
print(f"Status: {status_map.get(sol.get_status(), sol.get_status())}")
print()
routes = sol.get_route()
print("排程路由：")
print(routes.to_string())
print()
print(f"總目標值（WCT + 截止懲罰 + 機台偏好成本）: {sol.get_total_objective()}")
print()

# 分析結果
lot_names = {1: "lot_0", 2: "lot_1", 3: "lot_2", 4: "lot_3"}
machine_names = {0: "machine_0", 1: "machine_1"}
print("分析：")
deliveries = routes[routes["type"] == "Delivery"].to_pandas()
for _, row in deliveries.iterrows():
    lot = lot_names.get(int(row["location"]), f"loc{int(row['location'])}")
    machine = machine_names.get(int(row["truck_id"]), f"m{int(row['truck_id'])}")
    start = int(row["arrival_stamp"])
    lot_idx = int(row["location"]) - 1
    due = due_times[lot_idx]
    on_time = "（準時 ✓）" if due == INF or start <= due else f"（逾時！截止={due}）"
    pref = "（偏好機台 ✓）" if (row["truck_id"] == 0 and lot_idx <= 1) or (row["truck_id"] == 1 and lot_idx >= 2) else "（非偏好機台）"
    print(f"  {machine} 處理 {lot}：t={start} 開始  {on_time} {pref}")
