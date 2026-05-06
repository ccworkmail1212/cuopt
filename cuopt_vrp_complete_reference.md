# cuOpt VRP 演算法完整技術參考手冊

> 基於 `cuopt_latest/cpp/src/routing/` 原始碼分析  
> 版本：GitHub 最新（commit 7360166）

---

## 目錄

1. [目錄結構](#1-目錄結構)
2. [常數與配置](#2-常數與配置)
3. [Fitness 公式與維度系統](#3-fitness-公式與維度系統)
4. [整體演算法流程](#4-整體演算法流程)
5. [Phase 1：Island 初始生成](#5-phase-1island-初始生成)
6. [Phase 2：Working Loop](#6-phase-2working-loop)
7. [improve_population](#7-improve_population)
8. [improve_population_fixed_threshold（GA 核心）](#8-improve_population_fixed_threshold)
9. [recombine：交叉算子](#9-recombine交叉算子)
10. [adjust_weights：動態懲罰調整](#10-adjust_weights)
11. [族群管理（Population & Diversity）](#11-族群管理)
12. [GES：可行性引擎](#12-ges可行性引擎)
13. [局部搜索（lm.improve）](#13-局部搜索lmimprove)
14. [解的表示與費用計算](#14-解的表示與費用計算)
15. [節點維度正反向傳播](#15-節點維度正反向傳播)
16. [Adapter 層：生成與修復](#16-adapter-層生成與修復)
17. [排班相關特殊機制](#17-排班相關特殊機制)
18. [設計哲學總結](#18-設計哲學總結)

---

## 1. 目錄結構

```
routing/
├── diversity/                  ← 演算法主框架（多島 GA）
│   ├── diverse_solver.hpp      ← 核心：1368行，整體演算法
│   ├── diversity_config.hpp    ← 常數：島大小、族群大小
│   ├── population.hpp          ← 族群管理：Clearing Radius
│   ├── macros.hpp              ← NDIM=9，維度索引，apply_costs
│   └── injection_info.hpp      ← 外部解注入機制
│
├── ges/                        ← Guided Ejection Search
│   ├── guided_ejection_search.cu/cuh  ← GES 主迴圈
│   ├── ejection_pool.cuh       ← EP（彈出池）LIFO stack
│   ├── execute_insertion.cu/cuh ← 三步插入策略
│   ├── compute_fragment_ejections.cu/cuh ← Fragment 評估 kernel
│   ├── compute_delivery_insertions.cuh   ← PDP 插入邏輯
│   ├── squeeze.cu/cuh          ← 壓縮操作
│   ├── eject_until_feasible.cu ← 初始可行性強制
│   ├── found_solution.cuh      ← 候選解資料結構
│   └── lexicographic_search/   ← 連鎖移位搜索
│
├── local_search/               ← 局部搜索算子
│   ├── local_search.cu/cuh     ← 主協調器
│   ├── two_opt.cu              ← 邊交換
│   ├── sliding_window.cu       ← 視窗排列搜索
│   ├── sliding_tsp.cu          ← 單路線最優 TSP
│   ├── random_cross.cu         ← 隨機跨路線移動
│   ├── prize_collection.cu     ← Prize 收集移動
│   ├── breaks_insertion.cu     ← Break 插入
│   ├── compute_insertions.cu   ← 插入候選計算（IMPROVE/RANDOM/CROSS）
│   ├── perform_moves.cu        ← 移動執行
│   ├── cycle_finder/           ← 負環偵測
│   ├── move_candidates/        ← 候選移動資料結構
│   ├── vrp/                    ← VRP 專用移動
│   └── hvrp/                   ← 異質車隊（regret heuristics）
│
├── crossovers/                 ← 交叉算子
│   ├── srex_recombiner.hpp     ← SREX（集合覆蓋交換）
│   ├── eax_recombiner.hpp      ← EAX/AEAX（邊集合交換）
│   ├── ox_recombiner.cuh       ← OX（最短路 DP 劃分）
│   ├── inversion_recombiner.hpp ← IX（片段反轉）
│   ├── dispose.hpp             ← DISPOSE（移除路線）
│   ├── optimal_eax_cycles.cu/cuh ← EAX 環偵測
│   ├── set_covering.hpp        ← SREX 的集合覆蓋求解
│   └── ox_graph.hpp            ← OX 的圖結構
│
├── node/                       ← 節點維度計算（GPU kernel 內呼叫）
│   ├── node.cuh                ← 基礎模板
│   ├── time_node.cuh           ← 時間窗維度
│   ├── capacity_node.cuh       ← 容量維度
│   ├── break_node.cuh          ← Break 維度
│   ├── distance_node.cuh       ← 距離 objective
│   ├── tasks_node.cuh          ← 任務 variance objective
│   ├── service_time_node.cuh   ← 服務時間 variance
│   ├── prize_node.cuh          ← Prize objective
│   ├── vehicle_fixed_cost_node.cuh ← 固定車輛成本
│   └── mismatch_node.cuh       ← 不匹配成本
│
├── route/                      ← 路線層維度追蹤
│   ├── route.cuh               ← 路線模板
│   ├── dimensions_route.cuh    ← 多維度路線包裝
│   ├── time_route.cuh          ← 時間維度路線
│   ├── capacity_route.cuh      ← 容量維度路線
│   ├── tasks_route.cuh         ← 任務 variance 路線
│   └── ...（其他維度路線）
│
├── solution/                   ← 解的表示
│   ├── solution.cuh/cu         ← 核心解資料結構
│   ├── route_node_map.cuh      ← 節點 → 路線索引映射
│   ├── pool_allocator.cuh      ← GPU 記憶體池
│   └── solution_handle.cuh     ← GPU stream 管理
│
├── adapters/                   ← Host-Device 介面
│   ├── adapted_sol.cuh         ← Host 端解包裝（含相似度計算）
│   ├── adapted_generator.cu/cuh ← generate_solution / make_feasible
│   ├── adapted_modifier.cu/cuh  ← lm.improve / add_unserviced
│   └── assignment_adapter.cuh   ← 車輛分配介面
│
├── problem/                    ← 問題定義
│   ├── problem.cu/cuh          ← 問題資料結構
│   └── special_nodes.cuh       ← Break 等特殊節點
│
├── generator/                  ← 解初始化
│   └── generator.cu/hpp        ← GES 生成器包裝
│
└── 根目錄（高層介面）
    ├── solver.cu/hpp           ← 主 solver 介面
    ├── solve.cu                ← VRP 求解入口
    ├── ges_solver.cu/cuh       ← GES solver 包裝
    ├── dimensions.cuh          ← 維度類型（dim_t, objective_t）
    ├── structures.hpp          ← 公開資料結構
    └── vehicle_info.hpp        ← 車輛資訊
```

---

## 2. 常數與配置

### diversity_config.hpp

```cpp
enum class config_t : int { DEFAULT, CVRP, TSP };

struct diversity_config_t {
  // 每個 Island 最少解數
  min_island_size<DEFAULT>() = 3
  min_island_size<TSP>()     = 10
  min_island_size<CVRP>()    = population_size<CVRP>() / 2 = 8

  // 每個 Island 族群大小
  population_size<any>() = 16

  // CVRP 的 Island 總數
  island_size<CVRP>() = 5
}
```

### diverse_solver.hpp（頂部常數）

```cpp
constexpr int max_sol_per_population          = 32;
constexpr int default_reserve_population_size = 32;
constexpr int max_perturbation_reserve        = 10;
constexpr int min_perturbation_reserve        = 0;

// 每個 Island 最少生成時間（ms），依問題規模選擇
// 索引 = n_orders / 100（上限 10）
min_single_island_generation_time = {
  30/*n<100*/, 40/*<200*/, 50/*<300*/, 80/*<400*/,  90/*<500*/,
 100/*<600*/, 120/*<700*/, 180/*<800*/, 240/*<900*/, 260/*<1000*/, 300/*>1000*/
};

// 時間預算
first_sol_gen_time = min(time_limit × 0.3, 300s)   // 第 1 個解
sol_gen_time       = min(time_limit × 0.05, 60s)   // 後續每個解
                   × ges_time_fraction              // prize 問題時 = 0.1

// diversity_levels 建立（最多 8 層）
max = n_orders - 2
step_number = max(100, min(n_orders / 2, 200))
for 8 levels:
  diversity_levels.push(max / n_orders)
  step_lengths.push(step_number)
  if step_number > 149: step_number -= 50
  if level <= 0.55: break
  max -= (CVRP ? 8 : 32)
// 範例 n=200, VRP: levels=[0.99,0.83,0.67,0.51], steps=[150,150,100,100]
```

---

## 3. Fitness 公式與維度系統

### macros.hpp：9D 約束維度

```cpp
DIST             = 0   // 距離超限
TIME             = 1   // 時間窗違規量
CAP              = 2   // 容量超載量（支援多維容量）
PRIZE            = 3   // Prize 未收集懲罰
TASKS            = 4   // 任務數 variance
SERVICE_TIME     = 5   // 服務時間 variance
MISMATCH         = 6   // PDP pickup-delivery 不匹配
BREAK            = 7   // Break 未滿足量
VEHICLE_FIXED_COST = 8 // 車輛固定成本
NDIM = 9

using costs = std::array<double, NDIM>;
apply_costs(in, weights) = Σᵢ in[i] × weights[i]  // 9D 點積
```

### routing_structures.hpp：6 種 Objective

```cpp
enum class objective_t {
  COST,                        // 距離 × cost matrix
  TRAVEL_TIME,                 // 行駛時間（不含等待）
  VARIANCE_ROUTE_SIZE,         // 路線節點數 variance（≈ 工作量均衡）
  VARIANCE_ROUTE_SERVICE_TIME, // 服務時間 variance
  PRIZE,                       // 收集的 prize 總和（取負號）
  VEHICLE_FIXED_COST,          // 使用車輛的固定成本
  SIZE  // = 6
};
```

### 完整 Fitness 公式

```
fitness(解) = Σᵢ working_weights[i] × infeasibility_cost[i]   (9D，由 adjust_weights 動態調整)
            + Σⱼ obj_weights[j]     × objective_cost[j]        (6D，使用者設定，固定不變)
```

**兩類 weights 的關係：**
- `working_weights`：懲罰約束違規，由 `adjust_weights()` 動態調整
- `final_weights`（= obj_weights）：真正的優化目標，固定
- `adjust_weights` 只改 `working_weights[i]`，且上限 = `final_weights[i]`

### 初始 working_weights（ges_solver.cu）

| 維度 | 初始值 |
|------|--------|
| TIME | 10000 |
| CAP | 10000 |
| MISMATCH | 10000 |
| BREAK | 10000 |
| VEHICLE_FIXED_COST | 10000 |
| DIST | 100 |
| PRIZE / TASKS / SERVICE_TIME | 1000 |

---

## 4. 整體演算法流程

```
routing::Solve(data_model, settings)
    │ C API / Python wrapper
    ▼
solver_t::solve()
    │ 驗證輸入、判斷 VRP / PDP / TSP
    ▼
ges_solver_t::compute_ges_solution()
    │ 初始化 weights、建立 solve<> 實例
    ▼
solve::perform_search(routes_number, feasible_only)
    │
    ├─ run_ls_and_exit()  [若設定 CUOPT_LS_INPUT_PATH，直接 LS 後退出]
    │
    ├─ generate_from_scratch()           ← Phase 1
    │   └─ 若 reserve 退化或時間到 → 提早返回
    │
    ├─ start_reserve_threshold_adjustment()  [記錄 Phase 2 起始時間]
    │
    └─ run_working_loop()               ← Phase 2
        └─ 時間到 → 返回 reserve.best()
```

### Phase 1 vs Phase 2 的本質差異

Phase 1 和 Phase 2 使用**相同的內部機制**（GES + lm.improve + improve_population），差異在於**族群結構與互動方式**：

| | Phase 1 | Phase 2 |
|--|---------|---------|
| **外層驅動** | for Island（固定次數） | while time（時間驅動） |
| **族群結構** | 多個 Island 完全隔離 | 單一共用 reserve |
| **跨族群互動** | 無（直到 Phase 1 結束） | 每輪從 reserve 抽取、改完寫回 |
| **threshold 起點** | 固定從 index=3 開始 | 動態計算（依 working 解的多樣性） |
| **island_mode** | True（每層後檢查時間） | False（跑完所有層） |
| **adjust_weights** | 無 | 每輪執行 |
| **working 清空** | 無（島內持續累積） | 每輪清空重抽 |

**設計邏輯：**
- Phase 1 的 Island 隔離 = 強制多樣性，防止早熟收斂。若 Island 共享解，某個好解會快速擴散，三個 Island 很快收斂到同一局部最優。
- Phase 2 的合併 = 利用多樣性，讓不同 Island 的「基因片段」透過 GA 重組。Island 的使命在 Phase 1 結束時完成，繼續隔離反而浪費。

---

## 5. Phase 1：Island 初始生成

### generate_from_scratch()

```
reserve_population.max_solutions = 32
working_population.max_solutions = 32

generate_initial(target_vehicles_)
    │
    └─ 見下節

from_islands = load_sols_from_islands()  // 每個 Island 取最佳解

if initial_islands.size() > 1:
    threshold_index = find_initial_diversity(from_islands, avg=true)
    // 兩兩計算路線邊重疊率（相似度），取平均值
    // 找對應的 diversity_levels index
    reserve.threshold = max(0.8, diversity_levels[threshold_index])
else:
    reserve.threshold = 0.99

for island in initial_islands:
    reserve.add(island.best_feasible() if feasible else island.best())

if reserve.size() < 10:
    refill_reserve()  // GES + lm.improve 補充，時間上限 = min(5%剩餘, 20s)

// 注意：reserve 不只在這裡接收解，島內 improve_population 每個 threshold 層
// 開始前都會呼叫 p.add_solutions_to_island(reserve)，把島內「所有可行解」
// 同步到 reserve。實際進入 Phase 2 的解遠多於「每島 1 個」。
```

### reserve < 10 的原因

reserve 解不足 10 個通常由三個原因組合導致：

1. **Clearing Radius 雙層過濾**：
   - 第一層：島內 `add_solution()` 用島的 threshold 過濾（diversity_levels[當前層]）
   - 第二層：`add_solutions_to_island()` 同步到 reserve 時，reserve 用自己的 threshold（≥ 0.8）再過濾
   - reserve 的 threshold 通常比島內寬鬆，主要擋掉跨 Island 間幾乎相同的解

2. **可行解不夠多**：`add_solutions_to_island` 只同步可行解。若某 Island 時間內沒找到可行解，對 reserve 貢獻為 0。

3. **時間太短**：Island 在 60% 時限前被強制停止，可能只完成了 min_island_size 個初始解，island 內 GA 幾乎沒跑。
```

### generate_initial()：Island 生成詳情

**Island 數量計算：**
```
CVRP: islands_size = 5（固定）

VRP/PDP:
    generation_time_index = min(10, n_orders / 100)
    single_gen_time = min_single_island_generation_time[generation_time_index]
    islands_size = max(3, min(5, time_limit / (3 × single_gen_time)))

min_single_island_generation_time 查表（秒）：
  n_orders < 100  → 30s
  n_orders < 200  → 40s
  n_orders < 300  → 50s
  n_orders < 400  → 80s
  n_orders < 500  → 90s
  n_orders < 600  → 100s
  ...

實際例子（time_limit=60s）：幾乎永遠只有 3 個 Island（時間太短）
實際例子（time_limit=600s，n_orders<200）：min(5, 600/120)=5 → 5 個 Island

Island 生成停止條件：elapsed >= 60% × time_limit AND NOT first_gen
```

**Island 族群大小（hardcoded constexpr，使用者無法調整）：**
```
min_island_size（每個 Island 至少 GES 建幾個初始解）：
  VRP/PDP（DEFAULT）= 3
  CVRP              = 8（= population_size / 2）
  TSP               = 10

population_size（族群容量上限）= 16（所有類型相同）
```

**每個 Island 的時間預算：**
```
max_island_generation_time:
  第 1 個 Island = min(time_limit, 2000s)
  之後每個      = min(max(0, 60%budget - elapsed) × 0.4, 2000s)
```

**每個 Island 的流程：**
```
for i in range(min_island_size):  // VRP=3, CVRP=8

    ① 生成解：
       if first_gen AND routes_number == -1:
           g.generate_solution(time_limit=first_sol_gen_time)  // 30%
           target_vehicles_ = 解使用的車輛數  // 自動決定
       else:
           g.generate_solution(time_limit=sol_gen_time)  // 5%

       if NOT feasible: sol_gen_time += 20s  // 失敗則增加時限

    ② 立即 lm.improve(sol, final_weights, remaining_time)
       // 若 LS 把可行解搞壞 → final_weights[dim] × 10

    ③ island.add_solution(sol)
       reserve_population.add_solution(sol)  // 不等 Island 完成，立即加入

    if island_creation_timer.check(): break

④ improve_population(island, island_mode=true, start_index=3)
   // threshold 從 diversity_levels[3] 往 0 遞減
   // 每完成一層重算剩餘時間 = max(0, 60%budget - elapsed)
   // 超過則停止
```

---

## 6. Phase 2：Working Loop

### run_working_loop()：每輪步驟

```
improvement_timer = timer  // 用於島內改良的子計時器

WHILE NOT timer.check_time_limit():

  ① adjust_reserve_threshold()
     reserve_time_ratio = (elapsed - start) / (limit - start)
     reserve_time_ratio = reserve_time_ratio²  // 二次曲線
     reserve.threshold  = init_threshold + ratio² × (0.99 - init_threshold)
     // 效果：後期越來越寬鬆，接受更相似的解進入 reserve

  ② populate_working_vector()
     n = max(2, min(5, reserve.size() / 2))
     if elapsed < 70% × time_limit:
         get_n_random(n, tournament=true)   // 探索
     else:
         get_n_best(n)                      // 利用

  ③ threshold_index 設定
     if is_cvrp:
         threshold_index = 1  // 固定
     else:
         threshold_index = find_initial_diversity(working_vector, avg=false)
         // VRP 取最大相似度（而非平均）
         threshold_index = min(4, max(2, threshold_index))  // clamp
     working.threshold = diversity_levels[threshold_index]

  ④ populate_working_population()

  ⑤ [CVRP only] 額外生成 5 個解：
     single_gen_time = min(remaining × 0.05, 20s) × ges_time_fraction
     for i in 5:
         g.generate_solution(time=single_gen_time)
         lm.improve(final_weights)
         reserve.add(sol)    // 直接進 reserve
         working.add(sol)

  ⑥ if working.size() == 0: continue

  ⑦ best_before = working.best_feasible().cost if feasible else MAX

  ⑧ improve_population(working, island_mode=false, threshold_index)
     // 核心搜索，見第 7 節

  ⑨ best_found = working.best()
     if NOT best_found.is_feasible():
         lm.perturbate(best_found, final_weights, perturbation_count + 1)
         // 大幅擾動跳出局部最優（不影響 working 族群）

  ⑩ adjust_weights(best_before)  // 見第 10 節

  ⑪ add_working_to_reserve()
     if weights ≠ final_weights:
         for not_feasible in working:
             lm.improve(not_feasible, final_weights, ...)  // 用真實 weights 再改良
     working.add_solutions_to(reserve)

  ⑫ if NOT reserve.is_feasible():
         run_make_feasible()  // 從 reserve 取解，GES 急救，lm.improve

  ⑬ working.clear()
     if NOT best_found.is_feasible():
         working.add(best_found)  // 不可行的最佳解帶入下輪

  ⑭ working.change_weights(weights)  // 更新 working 的 weight

  ⑮ if reserve.size() < 5: refill_reserve()
```

---

## 7. improve_population

```
improve_population(p, island_mode, start_threshold_index, consider_expensive=true):

  if p.size() < 2: return

  WHILE start_threshold_index >= 0:

    valid_idx = min(start_threshold_index, len(step_lengths) - 1)
    p.threshold = diversity_levels[valid_idx]

    // 每層開始前先同步到 reserve
    p.add_solutions_to(reserve_population)

    // 重置可搜索路線標記
    p.best().reset_viable_of_problem()

    improve_population_fixed_threshold(
        p,
        max_iter    = step_lengths[valid_idx],  // 100~200
        threshold_idx = start_threshold_index,
        consider_expensive
    )

    // 若最佳解仍不可行，最後加強一次
    if NOT p.best().is_feasible():
        temp = p.best()
        lm.improve(temp, final_weights, remaining_time, run_cycle_finder=true)
        p.add_solution(temp)

    start_threshold_index--

    if island_mode:
        time_left = max(0, 60%budget - elapsed)
        improvement_timer = timer(time_left)
        if improvement_timer.check(): return  // Island 時間到

    if timer.check_time_limit(): return
```

---

## 8. improve_population_fixed_threshold

**GA 核心迴圈（Phase 1 & 2 共用）**

```
OUTER WHILE (improved=true 繼續，false 退出):
    k = max_iterations (100~200)
    improved = false

    INNER WHILE k-- > 0:

        if improvement_timer.check(): return
        if p.size() < 2: return  // 族群退化

        ① Tournament Select（CPU）
           p.get_two_random(temp_pair, tournament=true)
           // min(rand1, rand2) × 2 各取，選較優者
           cost_first  = temp_pair.first.get_cost(weights)
           cost_second = temp_pair.second.get_cost(weights)
           temp_pair.{first,second}.unset_routes_to_search()

        ② 決定使用哪些 recombiner
           run_expensive = consider_expensive AND start_threshold_idx <= 4
           run_cycle_finder = start_threshold_idx <= 1
           if VRP:
               run_expensive    = true  // 永遠用昂貴算子
               run_cycle_finder = true

        ③ recombine(temp_pair.first, temp_pair.second, guiding, run_expensive)
           // 見第 9 節，回傳 bool（成功/失敗）
           // 每個 recombiner 完成後 GES repair unserviced nodes

        ④ if success:
               offspring = (guiding==false ? first : second)
               if NOT feasible_only OR offspring.is_feasible():
                   lm.improve(offspring, weights, improvement_timer, run_cycle_finder)
                   // 最小化在此！Δfitness < 0 才 accept，GPU 並行評估
                   working_insertion_index = p.add_solution(offspring)

        temp_pair.{first,second}.set_routes_to_search()

        ⑤ 決策
           if working_insertion_index != -1 AND working_insertion_index <= 3:
               improved = true  // 進入前 4 名
               break            // 跳出內層，重置 k
```

### step_lengths（k 值）的計算

k 不是固定的，依訂單數與 threshold 層動態決定：

```cpp
// 初始化時計算（n_orders > 40 的情況）：
step_number = max(100, min(n_orders / 2, 200))

// 每個 threshold 層的 step_length：
step_lengths = [step_number, step_number-50, ...]  // 每層遞減 50，最低不低於 100
// 例如 n_orders=400：step_lengths = [200, 150, 100, 100, ...]
// 特殊：n_orders <= 40 時，step_lengths = [20]（小問題）
```

**越往後的 threshold 層 k 越小**，因為越嚴格的多樣性要求下大多數 offspring 會被 Clearing Radius 拒絕，給太多次是浪費。

### OUTER while 的三個終止條件

| 條件 | 程式碼 | 意義 |
|------|--------|------|
| k 次都未進 top 4 | `improved` 仍為 false | 當前 threshold 已飽和 |
| 時間到 | `improvement_timer.check_time_limit()` | 每次內層迭代開頭檢查 |
| 族群退化 | `p.current_size() < 2` | 族群槽位（index[1] 以後）少於 2 個，無法 tournament select |

> `current_size() = indices.size() - 1`，因為 `indices[0]` 是專門保留最佳可行解的槽位，不計入一般族群大小。族群退化通常是 Clearing Radius 過於嚴格，把大多數解踢掉導致。

**評估公式（嵌入 GPU kernel）：**
```
Δfitness = dot(working_weights[9D], Δinfeasibility)
         + dot(obj_weights[6D],     Δobjective)
= calculate_forward_all_and_delta()  ← HDI 函數，在 GPU kernel 內執行
accept if Δfitness < 0  → 梯度下降
```

---

## 9. recombine：交叉算子

### 算子選擇邏輯（diverse_solver.hpp:1094）

```
recombine_options:
  if is_cvrp:
      options = {SREX, IX}
  else:
      options = {OX}
      if a.routes > 1 AND b.routes > 1:  // 多路線
          options += {DISPOSE, SREX}
      elif PDP single route:
          options += {EAX, AEAX}
      if run_expensive:
          options += {EAX, AEAX}
          if NOT tsp: options += {IX}

從 options 均勻隨機選 1 個執行（非加權）
```

### SREX（Set Covering EXchange）

```
srex_recombiner.hpp

1. 找出 A 和 B 各自獨有的路線（different routes）
   if |ids_A| <= 1 OR |ids_B| <= 1: FAIL

2. Set Covering Problem 求解：
   選最少路線覆蓋所有節點
   target = min(|ids_A|, |ids_B|) 條路線

3. 貪婪分配：
   隨機選 guiding_route（來自 A 或 B）
   加入節點到 offspring（跳過已加入的）
   未覆蓋節點標為 unserviced

4. 替換路線；lm.add_unserviced_request() 插回未服務節點
```

### OX（Order Crossover，含 Bellman-Ford DP）

```
ox_recombiner.cuh

1. 抽取 genome（節點序列）
2. 從 B 取隨機片段，用 A 的順序填其餘
3. 建圖：edge_cost(i→j) for all i<j
   GPU kernel: calculate_edge_costs_kernel
4. Bellman-Ford DP：
   DP[k][j] = 用恰好 k 條路線走到節點 j 的最小 cost
   GPU kernel: bellman_ford_kernel
5. 回溯最優路線劃分
6. lm.make_cluster_order_feasible_request()
```

### EAX / AEAX（Edge Assembly Crossover）

```
eax_recombiner.hpp
constexpr int max_eax_cycle_length = 64

1. 建 E-set：A 有 B 沒有的邊 ∪ B 有 A 沒有的邊
2. 找環：交替使用 A/B 的邊（AEAX = 非對稱版本）
3. 選最接近 perfect_edges_number 的環
4. 積分：移除邊、建環、整合進解
5. lm.add_cycles_request() + make_cluster_order_feasible()
```

### IX（Inversion）

```
inversion_recombiner.hpp
route_size_limit = 60 nodes

1. lm.equalize_routes_and_nodes(a, b, skip_adding=true)
   讓兩解路線數相同
2. 計算每個節點在 B 中的位置（相對排序）
3. 用 mergesort 計算 A 相對 B 的反向數（inversion count）
4. 選反向數最多的片段（最多 5 條路線，總長 ≤ 60）
5. 依 B 的節點順序重排片段
```

### DISPOSE

```
dispose.hpp

1. if n_routes <= min_vehicles: FAIL
2. 隨機選一條路線刪除
3. 被移除路線的節點標為 removed_nodes
4. lm.add_selected_unserviced_requests(offspring, removed_nodes)
   → GES 嘗試插回
```

### 交叉後 GES repair（每個 recombiner 共通）

```
SREX/DISPOSE:
  lm.add_unserviced_request(offspring, weights)
    → populate_ep_with_unserved()
    → EP.random_shuffle()
    → ges.squeeze_all_ep()  // GPU 嘗試插回所有 EP 訂單

EAX/AEAX:
  lm.add_cycles_request(offspring, eax.cycles, weights)
  lm.make_cluster_order_feasible_request(offspring, weights)

OX:
  lm.make_cluster_order_feasible_request(a, weights)

if has_vehicle_breaks AND success:
  lm.squeeze_breaks(offspring, weights)  // Break 插入最佳化
```

---

## 10. adjust_weights

```
adjust_weights(best_before_improvement):

  best_found = working.best()
  cost_feasible = working.best_feasible().cost if feasible else MAX
  is_new_feasible_better = cost_feasible + ε < best_before

  // 決定調整係數
  CASE 1：best_before ≠ MAX AND best_found NOT feasible AND is_new_feasible_better
    // 舊有可行解，但改良後最佳解變不可行，且新可行更好
    adjust_coeff_tmp = uniform(0.99, 1.01)  // 幾乎不動
  CASE 2：其他情況
    adjust_coeff_tmp = adjust_coeff_weights + uniform(-0.04, +0.04)
    // adjust_coeff_weights 本身也更新：
    if adjust_coeff_weights < 1.1:
        adjust_coeff_weights = 1.06
    else:
        adjust_coeff_weights *= (0.8 + uniform(-0.04, +0.04))  // 0.76~1.84

  // 更新各維度 working_weights
  for i in 9:
      if best_found.infeasibility[i] == 0:   // 維度 i 可行
          if weights[i] > ε:
              weights[i] /= adjust_coeff_tmp  // 降低懲罰
      else:                                    // 維度 i 不可行
          if weights[i] < final_weights[i]:    // 不超過上限
              weights[i] *= adjust_coeff_tmp   // 提高懲罰
```

---

## 11. 族群管理

### 相似度計算（adapted_sol.cuh:252）

```cpp
calculate_similarity_radius_asymetric(A, B):
  common_edges = 0
  for i in range(n_orders):
      if succ_A[i] == succ_B[i] OR succ_A[i] == pred_B[i]:
          common_edges++
      elif is_cvrp AND i,succ[i] 在 B 同一路線:
          common_edges++
  common_edges *= 2  // 對稱化

  // 加計路線邊界
  for route_start in A: if B 的前後是 depot: common_edges++
  for route_start in B: if B 的前後是 depot: common_edges++

  max_diff = n_routes_A + n_routes_B + 2×(n_orders - depot_included)
  return common_edges / max_diff  // [0, 1]

// PDP/CVRP：取對稱平均
similarity(A, B) = 0.5 × (asymetric(A,B) + asymetric(B,A))
// VRP：直接用非對稱版本
```

### 族群插入邏輯（population.hpp:242）

```
add_solution(time, sol):
  cost = sol.get_cost(final_weights)

  // 快速拒絕
  if indices.size() == max AND cost >= worst_cost: return -1

  // 更新最佳可行解（特殊 slot 0）
  if sol.feasible AND cost < best_feasible_cost:
      solutions[0] = sol

  // 相似度檢查
  index = best_similar_index(sol)
  // 掃描 indices[1..]，找第一個 similarity(sol, existing) > threshold

  if index == max（無相似解）:
      if 族群已滿: 移除 indices 末尾（最差）
      insert_sorted_by_cost(sol, cost)
      return position

  elif cost < indices[index].cost（新解更好）:
      eradicate_similar(index)  // 移除所有相似解
      insert_sorted_by_cost(sol, cost)
      return position

  else: return -1  // 相似且不更好
```

### Clearing Radius 兩個族群

| 族群 | 初始 threshold | 變化 | 含義 |
|------|--------------|------|------|
| `reserve_population` | max(0.8, diversity_levels[idx]) | 二次增加 → 0.99 | 後期越寬鬆，接受更相似的解 |
| `working_population` | diversity_levels[-1]（最小） | 大→小逐層遞減 | 搜索從寬（探索）到嚴（聚焦） |

**Clearing Radius 語義：**  
`similarity(new, existing) > threshold` → 視為重複 → 拒絕或替換  
- threshold 高（0.99）= 幾乎一樣才算重複 = 寬鬆  
- threshold 低（0.8）= 80% 相似就算重複 = 嚴格

---

## 12. GES：可行性引擎

### 三種呼叫場景

| 場景 | 觸發 | 流程 |
|------|------|------|
| **建構初始解** | `generate_solution()` | random_init_routes → eject_until_feasible → init_EP → **fixed_route_loop** |
| **交叉後修復** | `add_unserviced_request()` | populate_ep_with_unserved → EP.shuffle → **squeeze_all_ep** |
| **強制修復** | `run_make_feasible()` | eject_until_feasible → populate_ep → random_LS×1 → **fixed_route_loop** |

### GES 關鍵常數（guided_ejection_search.cuh）

```cpp
constexpr int shuffle_interval         = 20;   // 每 20 次 失敗洗牌 EP
constexpr int eject_new_route_threshold = 100;  // 連續 100 次失敗停止（minimize_routes 模式）
constexpr auto insertion_rate          = 0.003; // 多插入時，0.3% 的請求數
```

### p_scores：引導機制

```
初始值：VRP = 1，PDP = 只有 pickup = 1（delivery = 0）

每次插入失敗：incr_p_scores<<<1,1>>>(request)  // request.p_score += 1

使用：在 execute_best_insertion_ejection_solution 中
      選擇踢出的 fragment 時，偏好踢出 p_score 低的（容易安置）
      高 p_score 的訂單保留在路線裡（難以安置，讓它留著）
```

### GES 主迴圈（guided_ejection_search_loop）

```
iteration_limit = 500000
if minimize_routes:
    iteration_limit = min(500000, N² / K)

n_insertions = max(1, floor(num_requests × 0.003))
n_insertions = min(n_routes, n_insertions)

WHILE EP.size() > desired_ep_size:
    if time_limit OR iteration_limit: return false

    // 多插入路徑（n_insertions > 1）
    if n_insertions > 1:
        n_insertions = try_multiple_feasible_insertions(n_insertions, perturb=true)
        continue

    request = EP.pop()  // LIFO

    ── Step 1：try_single_insert_with_perturbation ──
    perturbation_count = max(1, min(8, 100/n_routes))
    for i in perturbation_count:
        run_random_local_search()  // 擾動路線
    n_found = find_single_insertion(request)
        // GPU kernel: get_all_feasible_insertion<<<n_routes, 64>>>
        // 評估所有插入位置，隨機選一個可行的執行
    if found: continue

    ── 失敗 ──
    incr_p_scores(request)

    ── Step 2：execute_best_insertion_ejection_solution ──
    for fragment_size in [1..10 step fragment_step]:
        GPU kernel: kernel_get_best_insertion_ejection_solution
            <<<n_requests × fragment_step, 64>>>
        選 p_score 最低的 fragment 踢出，插入 request
    if found: EP.index_ += deleted_frag; continue

    ── Step 3：run_lexicographic_search ──
    k_max = 5 (route>30), 4(>50), 3(>100), 2(>200)
    GPU kernel: lexicographic_search
    找 p_score 加總最小的 k 步連鎖移位序列
    if found: execute_lexico_move; continue

    ── 全部失敗 ──
    consecutive_ejection_failure++
    if failure % 20 == 0:
        try_squeeze or shuffle_pool
    EP.push_back_last()
```

---

## 13. 局部搜索（lm.improve）

### 算子執行順序（local_search.cu）

```
run_best_local_search(sol, consider_unserviced, time_limit, run_cycle_finder):

  // 異質車隊：先做車輛分配
  if heterogeneous_fleet:
      hvrp::vehicle_assignment(sol)

  while iterations < iter_limit:

    extract_nodes_to_search(sol)  // 採樣搜索節點（VRP 特有）
    calculate_route_compatibility(sol)
    // ← GPU kernel<<<n_routes×n_requests, 128>>>
    // 計算每個 (route, request) 的相容性

    while true:
      // ─ 滑動視窗 ─
      if run_sliding_search(sol): continue

      // ─ Prize 收集 ─
      if has_prize AND run_collect_prizes(sol): continue

      // ─ Break 移動 ─
      if has_breaks AND perform_break_moves(sol): continue

      break

    // ─ Cycle Finder（多路線改良）─
    if n_routes >= 2 AND run_cycle_finder:
        find_insertions(sol, IMPROVE)
        find_best_negative_cycles(small_finder, big_finder)
        if found: apply_cycle_moves()
```

### 滑動視窗（sliding_window.cu）

```
window_sizes: 3, 4, 5, 6
permutations: 3!=6, 4!=24, 5!=120, 6!=720
max_range_size = 200 nodes

GPU kernel（block per window position）：
  extract window[start, start + window_size)
  for each valid permutation:
    check PDP order constraints
    evaluate cost_combine(prev, permuted_window, next)
    update if delta < best
```

### 2-Opt（two_opt.cu）

```
GPU kernel: find_two_opt_moves<<<sampled_nodes, 128~256>>>
  每個 block = 一個採樣節點
  嘗試所有 (first, second) 對
  evaluate_fragment()：把 [first+1..second] 反轉後算 cost delta
  top_k_candidates = 64（每 block 保留前 64 個改善）
```

### 隨機跨路線（random_cross.cu）

```
calculate_route_compatibility()  // 相容性矩陣
find_insertions(RANDOM mode):
  // 若候選 > 閾值，隨機洗牌後取前幾個（降低貪婪性）

populate_random_moves():
  fill_random_route_pair_moves<<<n_orders, 256>>>
  sort by route pair index
  pick_random_move_per_route_pair()
  select_random_route_pairs()
  perform_moves()
```

### 三種搜索模式（compute_insertions.cu）

| 模式 | 行為 | 使用場景 |
|------|------|---------|
| IMPROVE | 只接受改善，按 delta 排序 | 島內改良、working loop |
| RANDOM | 洗牌候選，隨機選 | Phase A 擾動 |
| CROSS | 跨路線最佳插入 | 跨路線重定位 |

---

## 14. 解的表示與費用計算

### solution_t（solution.cuh）

```cpp
struct solution_t {
  // CPU 端
  std::vector<route_t<>> routes;         // 路線列表
  i_t n_routes;
  std::vector<i_t> route_id_to_idx;

  // GPU 端（RMM device memory）
  device_uvector<route_t::view_t> routes_view;
  route_node_map_t route_node_map;
    // route_id_per_node[node]      → 所在路線
    // intra_route_idx_per_node[node] → 路線內位置

  device_scalar<infeasible_cost_t> infeasibility_cost;  // 9D
  device_scalar<objective_cost_t>  objective_cost;      // 6D
  device_scalar<i_t>  d_sol_found;         // 全局解旗標
  device_uvector<i_t> d_lock_per_route;    // 路線 spinlock
  device_uvector<i_t> routes_to_copy;      // 變動路線 bitmask
  device_scalar<i_t>  n_infeasible_routes; // 違規路線計數
};
```

### route_t（route.cuh）

```cpp
struct route_t {
  dimensions_route_t dimensions;  // 所有維度的節點序列
    // dimensions.requests.node_info[0..n_nodes-1]
    // 序列：[start_depot] [service₁] ... [serviceₙ] [end_depot]

  device_scalar<i_t> route_id;
  device_scalar<i_t> vehicle_id;
  device_scalar<i_t> n_nodes;
  device_scalar<infeasible_cost_t> infeasibility_cost;
  device_scalar<objective_cost_t>  objective_cost;

  const fleet_info_t* fleet_info_ptr;  // 車輛約束

  // 核心操作
  insert_node() / parallel_insert_node()
  eject_node()  / parallel_eject_node()
  compute_forward()   // 從頭到尾正向傳播
  compute_backward()  // 從尾到頭反向傳播
};
```

### 費用增量計算

```
node_t::calculate_forward_all_and_delta():
  for 每個啟用的維度（time/capacity/break...）:
      arc_value = 距離/時間矩陣查詢
      calculate_forward() 更新 forward state
      計算 Δinfeasibility_cost[9D]
      計算 Δobjective_cost[6D]
  return Δfitness = dot(working_weights, Δinfeasibility)
                  + dot(obj_weights, Δobjective)
```

---

## 15. 節點維度正反向傳播

### 時間維度（time_node.cuh）

```cpp
// 正向傳播
void calculate_forward(time_node_t& next, double time_between):
  next.departure_forward = departure_forward + time_between

  if next.departure_forward < window_start:
      next.departure_forward = window_start  // 等待
  elif next.departure_forward > window_end:
      next.excess_forward += (departure - window_end)  // 違規
      next.departure_forward = window_end

  next.transit_time_forward = transit_time_forward + time_between

  // 最遲合理到達（turn-around feasibility）
  next.latest_arrival_forward = latest_arrival_forward + time_between
  if latest_arrival < window_start:
      wait = window_start - latest_arrival
      next.unavoidable_wait_forward += wait
  elif latest_arrival > window_end:
      next.latest_arrival_forward = window_end

// 反向傳播（對稱）：excess_backward, earliest_arrival_backward

// 組合可行性（fragment check）
static combine(prev, next, vehicle_info, time_between):
  arrival_f = prev.departure_forward + time_between
  excess = prev.excess_forward + next.excess_backward
         + max(0, arrival_f - next.departure_backward)
         + max(0, total_time - vehicle_info.max_time)
  return excess  // 0 = 可行
```

### 容量維度（capacity_node.cuh，支援多維）

```cpp
// 正向
next.gathered[i]    = gathered[i] + next.demand[i]       // 累積載重
next.max_to_node[i] = max(next.gathered[i], max_to_node[i])  // 峰值

// 反向
prev.max_after[i] = max(0, max(prev.demand[i], prev.demand[i] + max_after[i]))

// 組合超載量
route_peak = max(prev.max_to_node[i], prev.gathered[i] + next.max_after[i])
excess[i]  = max(0, route_peak - vehicle_capacity[i])
```

### Break 維度（break_node.cuh）

```cpp
// 正向
next.breaks_forward = breaks_forward + breaks_in_between

// 違規量
inf_cost[BREAK] = max(0, total_breaks - vehicle.num_breaks())
// total_breaks = breaks_forward + breaks_backward
```

---

## 16. Adapter 層：生成與修復

### generate_solution()（adapted_generator.cu）

```
if problem.is_tsp:
    generate_tsp_solution()  // 隨機排列，單路線

else:
    if run_route_minimizer (desired_vehicles 未指定):
        ges.construct_feasible_solution()  // 從零建構
        ges.route_minimizer_loop()         // 逐步減少路線數
    else:
        sol.clear_solution(desired_vehicle_ids)
        sol.random_init_routes()       // 隨機分配
        sol.compute_initial_data()
        sol.eject_until_feasible()     // 踢出違規
        ges.init_ejection_pool()       // 收集 EP
        ges.fixed_route_loop()         // 清空 EP
        if has_breaks: ges.try_squeeze_breaks_feasible()

    ges.repair_empty_routes()
    sol.populate_host_data(true)
```

### make_feasible()（adapted_generator.cu）

```
ges.set_solution_ptr(&sol, clear_scores=true)
ges.start_timer(now, time_limit)
sol.eject_until_feasible(add_slack=true)   // 踢出違規節點至 EP
sol.populate_ep_with_unserved(ges.EP)
for perturbation_count=1: ls.run_random_local_search(sol)  // 擾動
ges.fixed_route_loop()                    // 清空 EP
ges.try_squeeze_breaks_feasible()         // Break 補救
return sol.is_feasible()
```

### add_unserviced_request()（adapted_modifier.cu）

```
resource.ges.set_solution_ptr(&sol)
sol.populate_ep_with_unserved(ges.EP)  // 收集 unserviced
ges.EP.random_shuffle()               // 打亂順序
ges.squeeze_all_ep()                  // 輕量 GES，嘗試全部插回
// squeeze = greedy_insert(all=true)，純插入不踢人
```

---

## 17. 排班相關特殊機制

### Break Scheduling

**問題定義：**
- 每輛車有若干 mandatory break（休息時間窗）
- Break 節點是特殊的路線節點（node_type_t::BREAK）
- `vehicle.num_breaks()` = 必須執行的 break 數

**維度追蹤（break_node.cuh）：**
```cpp
breaks_forward  // 路線從頭到此的 break 累積數
breaks_backward // 路線從尾到此的 break 累積數
// 違規：total_breaks < num_breaks → inf_cost[BREAK] 增加
```

**Break 插入最佳化：**
- `breaks_insertion.cu`：把 break 插入路線的最優位置
- `squeeze.cu`：把違規的 break 節點塞入空隙

### 時間窗對排班的影響

每個訂單（班次需求）設定 `earliest = latest = 指定日期`（time window = 1天）。這迫使車輛（員工）在指定日期服務該訂單，等同於「固定班次」。

### VARIANCE_ROUTE_SIZE 對休假公平的作用

```
tasks_node.cuh:
  obj_cost[VARIANCE_ROUTE_SIZE] = diff * diff
  where diff = route_service_nodes - avg_service_nodes_per_route

最小化路線間節點數的差異 = 最小化各員工工作天數差異
= 休假公平性的代理指標
```

---

## 18. 設計哲學總結

### 1. GES / LS 明確分工
- **GES**：唯一職責 = 不可行 → 可行（EP 清空即可行）
- **LS（lm.improve）**：唯一職責 = 可行域內最小化 fitness
- 兩者不越界，確保演算法的結構清晰

### 2. Memetic Algorithm
每個 offspring recombine 後**立刻** `lm.improve()`，進族群前已達局部最優。不讓未優化的解競爭。

### 3. 評估嵌入 GPU kernel
`calculate_forward_all_and_delta()` 在 GPU kernel 內執行（HDI 函數），評估即搜索，搜索即評估。不是獨立的 fitness 函數呼叫。

### 4. 動態 Fitness Landscape
`adjust_weights` 讓 working_weights 隨時間變動：
- 早期：高懲罰 → 逼向可行域
- 找到可行解後：降低懲罰 → LS 真正最小化 objective

搜索的不是固定的山，是會移動的山。

### 5. Clearing Radius 雙向調節
- `working.threshold`：大→小（搜索從寬到嚴，探索→聚焦）
- `reserve.threshold`：小→大（後期接受更相似的優解）

### 6. p_score 引導 GES
記憶每個訂單的「被卡住次數」。優先踢出容易安置的訂單（低 p_score），保護難以安置的訂單留在路線裡。

### 7. 70/30 探索利用切換
- 前 70% 時間：random 取解 → 廣泛探索
- 後 30% 時間：取 best 解 → 聚焦改良

### 8. Island 獨立搜索再匯聚
多個 Island 獨立演化（避免早熟），根據 Island 解的實際多樣性自動設定 reserve 的篩選嚴格度。

### 核心算法對照表

| 超啟發式步驟 | cuOpt 實作 | 位置 |
|------------|-----------|------|
| 初始化 | GES fixed_route_loop | adapted_generator.cu |
| 交換（產生新解） | recombine() 6 種算子 + GES repair | diverse_solver.hpp |
| 評估 | calculate_forward_all_and_delta()（GPU） | node/*.cuh |
| 決策 | p.add_solution() + Clearing Radius | population.hpp |
| 局部改良 | lm.improve() 2-opt/sliding/cross | local_search/*.cu |
