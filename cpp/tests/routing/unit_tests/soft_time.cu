/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * Unit tests for the SOFT_TIME dimension (weighted completion time objective).
 *
 * Problem mapping: orders = jobs, vehicles = machines/tools.
 *   - order_weight: per-order priority weight
 *   - service_time: per-order processing time on the vehicle
 *   - transit time: idle time mandated between orders (travel time matrix)
 *   - WCT = sum_k( order_weight[k] * completion_time[k] )
 *   - completion_time[k] = arrival_at_k + service_time[k]
 *
 * Test setup mirrors the hand-worked example in the code review:
 *   3 locations: 0=depot, 1=order_0, 2=order_1
 *   Transit times: t(0->1)=1, t(1->2)=2, t(0->2)=3  (and symmetric)
 *   Vehicle starts and ends at depot (loc 0), earliest = 0.
 */

#include <cuopt/routing/solve.hpp>
#include <routing/utilities/check_constraints.hpp>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

// Transit time matrix for 3 locations (row-major, 3x3):
//         depot  order_0  order_1
//  depot [  0      1        3    ]
//  order_0 [1      0        2    ]
//  order_1 [3      2        0    ]
static const std::vector<float> k_transit_times = {0, 1, 3, 1, 0, 2, 3, 2, 0};

// Uniform cost matrix so routing cost does not influence lot ordering.
static const std::vector<float> k_cost_matrix = {0, 1, 1, 1, 0, 1, 1, 1, 0};

/**
 * TEST 1 — correct ordering and WCT value
 *
 * orders:     order_0 (w=2, s=3),  order_1 (w=1, s=4)
 *
 * Route A: depot -> order_0 -> order_1 -> depot
 *   C_0 = 0 + t(0->1) + s_0 = 0 + 1 + 3 = 4
 *   C_1 = 4 + t(1->2) + s_1 = 4 + 2 + 4 = 10
 *   WCT = 2*4 + 1*10 = 18   <-- optimal
 *
 * Route B: depot -> order_1 -> order_0 -> depot
 *   C_1 = 0 + t(0->2) + s_1 = 0 + 3 + 4 = 7
 *   C_0 = 7 + t(2->1) + s_0 = 7 + 2 + 3 = 12
 *   WCT = 1*7 + 2*12 = 31
 *
 * Expected: solver picks Route A; WCT reported == 18.
 */
TEST(soft_time, correct_order_and_wct)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<int> order_locations = {1, 2};  // order_0 at loc 1, order_1 at loc 2
  std::vector<int> service_times   = {3, 4};  // s_0=3, s_1=4
  std::vector<int> order_weights   = {2, 1};  // w_0=2, w_1=1

  auto v_cost_matrix         = cuopt::device_copy(k_cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(k_transit_times, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);

  // 3 locations, 1 vehicle, 2 orders
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(2);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  // Verify route order: order_0 first, order_1 second
  ASSERT_EQ(host_route.route[1], 0);  // position 1 = order 0
  ASSERT_EQ(host_route.route[2], 1);  // position 2 = order 1

  // Verify reported WCT
  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 18.0, 1e-3);
}

/**
 * TEST 2 — higher weight on the faster order pulls it first
 *
 * Same transit times, but now order_1 is fast (s=1) with high weight (w=3),
 * and order_0 is slow (s=4) with low weight (w=1).
 *
 * Route A: depot -> order_0 -> order_1 -> depot
 *   C_0 = 0 + 1 + 4 = 5
 *   C_1 = 5 + 2 + 1 = 8
 *   WCT = 1*5 + 3*8 = 29
 *
 * Route B: depot -> order_1 -> order_0 -> depot
 *   C_1 = 0 + 3 + 1 = 4
 *   C_0 = 4 + 2 + 4 = 10
 *   WCT = 3*4 + 1*10 = 22   <-- optimal
 *
 * Expected: solver picks Route B; WCT reported == 22.
 */
TEST(soft_time, high_weight_order_served_first)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<int> order_locations = {1, 2};  // order_0 at loc 1, order_1 at loc 2
  std::vector<int> service_times   = {4, 1};  // s_0=4, s_1=1
  std::vector<int> order_weights   = {1, 3};  // w_0=1, w_1=3

  auto v_cost_matrix         = cuopt::device_copy(k_cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(k_transit_times, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(2);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  // Verify route order: order_1 (high weight, fast) first, order_0 second
  ASSERT_EQ(host_route.route[1], 1);  // position 1 = order 1
  ASSERT_EQ(host_route.route[2], 0);  // position 2 = order 0

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 22.0, 1e-3);
}

/**
 * TEST 3 — 10 orders, 4 vehicles, consistency check
 *
 * 11 locations: 0=depot, 1..10=order locations (one order per location).
 * Transit time matrix: 0 on diagonal, 1 everywhere else (uniform 1-unit transit).
 * 4 vehicles, 10 orders.
 *
 * Service times:  {2, 5, 1, 3, 2, 4, 3, 1, 2, 3}  (orders 0..9)
 * Order weights:  {3, 1, 4, 2, 5, 1, 3, 2, 4, 2}  (orders 0..9)
 *
 * No hand-computed optimal is needed. The test verifies:
 *   1. Solver returns SUCCESS and serves all 10 orders.
 *   2. The WCT reported by the solver equals the WCT computed by walking the
 *      flat route array independently (consistency check).
 *
 * Independent WCT walk:
 *   For each non-depot node i in the route array:
 *     order_id       = route[i]
 *     vehicle        = truck_id[i]
 *     completion[v]  = vehicle_completion[v] + transit(1) + service_times[order_id]
 *     wct           += order_weights[order_id] * completion[v]
 *     vehicle_completion[v] = completion[v]
 *   (transit is always 1 because all off-diagonal entries in the transit matrix are 1)
 */
TEST(soft_time, ten_orders_four_vehicles_consistency)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  const int n_locations = 11;
  const int n_vehicles  = 4;
  const int n_orders    = 10;

  // 11x11 transit matrix: 0 on diagonal, 1 elsewhere (used for arrival-time computation).
  // Cost matrix uses tiny values (0.001) so routing cost is negligible relative to WCT,
  // letting WCT dominate the search objective without triggering degenerate zero-cost routing.
  std::vector<float> transit_matrix(n_locations * n_locations);
  std::vector<float> cost_matrix(n_locations * n_locations);
  for (int i = 0; i < n_locations; i++) {
    for (int j = 0; j < n_locations; j++) {
      transit_matrix[i * n_locations + j] = (i == j) ? 0.f : 1.f;
      cost_matrix[i * n_locations + j]    = (i == j) ? 0.f : 0.001f;
    }
  }

  // order i is at location i+1
  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> service_times   = {2, 5, 1, 3, 2, 4, 3, 1, 2, 3};
  std::vector<int> order_weights   = {3, 1, 4, 2, 5, 1, 3, 2, 4, 2};

  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(transit_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, n_locations, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());
  // Force solver to use all vehicles (cuOpt minimizes vehicle count by default,
  // but WCT optimization benefits from parallel execution across all vehicles).
  data_model.set_min_vehicles(n_vehicles);

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(5);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  // All orders must be served
  ASSERT_EQ(host_route.unserviced_nodes.size(), 0u);

  // Independent WCT computation: walk flat route array
  std::vector<double> vehicle_completion(n_vehicles, 0.0);
  double computed_wct = 0.0;

  printf("[route] node_type | truck | order | completion | wct_contrib\n");
  for (int i = 0; i < static_cast<int>(host_route.route.size()); i++) {
    if (host_route.node_types[i] == 0) {
      printf("[route]  depot     | v%-4d |\n", host_route.truck_id[i]);
      continue;
    }
    int order_id = host_route.route[i];
    int v        = host_route.truck_id[i];
    // transit is always 1 (uniform off-diagonal matrix)
    double completion = vehicle_completion[v] + 1.0 + service_times[order_id];
    double contrib    = order_weights[order_id] * completion;
    computed_wct += contrib;
    vehicle_completion[v] = completion;
    printf("[route]  service   | v%-4d | order%-2d | %10.1f | %10.1f\n",
           v,
           order_id,
           completion,
           contrib);
  }

  printf("[ten_orders_four_vehicles] computed_wct = %.2f\n", computed_wct);

  // Reported WCT must match independently computed WCT
  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  printf("[ten_orders_four_vehicles] solver WCT   = %.2f\n",
         objectives.at(objective_t::WEIGHTED_COMPLETION_TIME));
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), computed_wct, 1e-3);
}

/**
 * TEST 4 — due_time: feasible solution when due times are generous
 *
 * Same 2-order setup as TEST 1 but with loose due time constraints.
 *   order_0: due_time = 100  (start by t=100; transit 1 => start ≈ 1; always feasible)
 *   order_1: due_time = 100
 *
 * The solver must still pick the WCT-optimal order (order_0 first) and
 * the reported solution must be feasible (no due time violations).
 */
TEST(soft_time, due_time_feasible_loose_constraints)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<int> order_locations = {1, 2};
  std::vector<int> service_times   = {3, 4};
  std::vector<int> order_weights   = {2, 1};
  std::vector<int> due_times       = {100, 100};  // very loose — no violation expected

  auto v_cost_matrix         = cuopt::device_copy(k_cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(k_transit_times, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);
  auto v_due_times           = cuopt::device_copy(due_times, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());
  data_model.set_order_due_times(v_due_times.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(2);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);

  // Optimal WCT order still holds: order_0 first (weight 2, faster completion)
  ASSERT_EQ(host_route.route[1], 0);
  ASSERT_EQ(host_route.route[2], 1);

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 18.0, 1e-3);
}

/**
 * TEST 5 — due_time: non-trivial tight deadlines on the 10-order / 4-vehicle problem
 *
 * Same problem as TEST 3, but with tight due time constraints on the two
 * low-weight orders (order1, order5) that the unconstrained optimal schedules last:
 *
 *   Unconstrained optimal (from TEST 3):
 *     order1 (s=5, w=1): 3rd on its vehicle, starts at t=6  => violates due_time=4
 *     order5 (s=4, w=1): 3rd on its vehicle, starts at t=7  => violates due_time=4
 *
 *   Structural constraint: an order in the 3rd (or later) slot always starts at
 *     t >= 1 + s_prev1 + 1 + s_prev2 + 1 = 3 + s_prev1 + s_prev2 >= 5,
 *   so setting due_time=4 guarantees these orders cannot remain in 3rd position.
 *   They must be moved to 1st or 2nd position (2nd only after an order with s<=2).
 *
 * The test verifies:
 *   1. Solver returns a valid solution with all orders served.
 *   2. Every order with a finite due_time actually starts within its deadline
 *      (independently verified by walking the flat route array).
 *   3. Reported WCT matches independently computed WCT (consistency check).
 */
TEST(soft_time, due_time_nontrivial_ten_orders_four_vehicles)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  const int n_locations = 11;
  const int n_vehicles  = 4;
  const int n_orders    = 10;

  std::vector<float> transit_matrix(n_locations * n_locations);
  std::vector<float> cost_matrix(n_locations * n_locations);
  for (int i = 0; i < n_locations; i++) {
    for (int j = 0; j < n_locations; j++) {
      transit_matrix[i * n_locations + j] = (i == j) ? 0.f : 1.f;
      cost_matrix[i * n_locations + j]    = (i == j) ? 0.f : 0.001f;
    }
  }

  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> service_times   = {2, 5, 1, 3, 2, 4, 3, 1, 2, 3};
  std::vector<int> order_weights   = {3, 1, 4, 2, 5, 1, 3, 2, 4, 2};
  // Tight due times only on order1 and order5 (low-weight orders the unconstrained
  // optimal schedules last at t=6 and t=7 respectively).  All others loose.
  std::vector<int> due_times = {100, 4, 100, 100, 100, 4, 100, 100, 100, 100};

  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(transit_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);
  auto v_due_times           = cuopt::device_copy(due_times, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, n_locations, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());
  data_model.set_order_due_times(v_due_times.data());
  data_model.set_min_vehicles(n_vehicles);

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(5);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  ASSERT_EQ(host_route.unserviced_nodes.size(), 0u);

  // Walk the flat route array: compute each order's actual start time and verify
  // all due_time constraints are satisfied, while also computing WCT.
  std::vector<double> tool_completion(n_vehicles, 0.0);
  double computed_wct = 0.0;

  printf("[due_time_nontrivial] node_type | truck | order | start | completion\n");
  for (int i = 0; i < static_cast<int>(host_route.route.size()); i++) {
    if (host_route.node_types[i] == 0) {
      printf("[due_time_nontrivial]  depot     | v%d\n", host_route.truck_id[i]);
      continue;
    }
    int order_id      = host_route.route[i];
    int v             = host_route.truck_id[i];
    double start      = tool_completion[v] + 1.0;  // transit always 1
    double completion = start + service_times[order_id];
    computed_wct += order_weights[order_id] * completion;
    tool_completion[v] = completion;

    printf("[due_time_nontrivial]  service   | v%d    | order%-2d | %5.1f | %10.1f\n",
           v,
           order_id,
           start,
           completion);

    // Verify due_time constraint for constrained orders
    if (due_times[order_id] < 100.) {
      EXPECT_LE(start, due_times[order_id]) << "order" << order_id << " starts at t=" << start
                                            << " but due_time=" << due_times[order_id];
    }
  }

  printf("[due_time_nontrivial] computed_wct = %.2f\n", computed_wct);

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  printf("[due_time_nontrivial] solver WCT   = %.2f\n",
         objectives.at(objective_t::WEIGHTED_COMPLETION_TIME));
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), computed_wct, 1e-3);
}

/**
 * TEST 6 — arrival time + due_time: 10 orders / 4 vehicles with both constraint types
 *
 * Builds on TEST 5 (due_time_nontrivial) by adding order arrival-time constraints.
 * Same 10-order / 4-vehicle setup; same due time constraints on order1 and order5;
 * additionally:
 *
 *   order4 (w=5, s=2): earliest_time = 4
 *     Unconstrained optimal schedules order4 first on its vehicle (start=1).
 *     With earliest_time=4 the vehicle must idle or put a short order before it.
 *
 *   order2 (w=4, s=1): earliest_time = 3
 *     Unconstrained optimal schedules order2 first on its vehicle (start=1).
 *     With earliest_time=3 the vehicle must idle or put a very short order before it.
 *
 * The test verifies:
 *   1. Solver returns SUCCESS with all orders served.
 *   2. Every order with a due_time starts within its deadline (start <= due_time).
 *   3. Every order with an earliest_time starts no earlier than that time
 *      (start >= earliest_time).
 *   4. Reported WCT == independently computed WCT (consistency check).
 */
TEST(soft_time, arrival_time_nontrivial_ten_orders_four_vehicles)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  const int n_locations = 11;
  const int n_vehicles  = 4;
  const int n_orders    = 10;

  std::vector<float> transit_matrix(n_locations * n_locations);
  std::vector<float> cost_matrix(n_locations * n_locations);
  for (int i = 0; i < n_locations; i++) {
    for (int j = 0; j < n_locations; j++) {
      transit_matrix[i * n_locations + j] = (i == j) ? 0.f : 1.f;
      cost_matrix[i * n_locations + j]    = (i == j) ? 0.f : 0.001f;
    }
  }

  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> service_times   = {2, 5, 1, 3, 2, 4, 3, 1, 2, 3};
  std::vector<int> order_weights   = {3, 1, 4, 2, 5, 1, 3, 2, 4, 2};
  // Due times: same tight deadlines as TEST 5
  std::vector<int> due_times = {100, 4, 100, 100, 100, 4, 100, 100, 100, 100};
  // Arrival times: order4 arrives at t=4, order2 arrives at t=3; others immediate.
  std::vector<int> earliest = {0, 0, 3, 0, 4, 0, 0, 0, 0, 0};
  std::vector<int> latest = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000};

  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(transit_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);
  auto v_due_times           = cuopt::device_copy(due_times, stream);
  auto v_earliest            = cuopt::device_copy(earliest, stream);
  auto v_latest              = cuopt::device_copy(latest, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, n_locations, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());
  data_model.set_order_due_times(v_due_times.data());
  data_model.set_order_time_windows(v_earliest.data(), v_latest.data());
  data_model.set_min_vehicles(n_vehicles);

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(5);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  ASSERT_EQ(host_route.unserviced_nodes.size(), 0u);

  // Walk the flat route array: verify both constraint types and compute WCT.
  std::vector<double> tool_completion(n_vehicles, 0.0);
  double computed_wct = 0.0;

  printf("[arrival_nontrivial] node_type | truck | order | start | completion\n");
  for (int i = 0; i < static_cast<int>(host_route.route.size()); i++) {
    if (host_route.node_types[i] == 0) {
      printf("[arrival_nontrivial]  depot     | v%d\n", host_route.truck_id[i]);
      continue;
    }
    int order_id      = host_route.route[i];
    int v             = host_route.truck_id[i];
    double start      = std::max(tool_completion[v] + 1.0, static_cast<double>(earliest[order_id]));
    double completion = start + service_times[order_id];
    computed_wct += order_weights[order_id] * completion;
    tool_completion[v] = completion;

    printf("[arrival_nontrivial]  service   | v%d    | order%-2d | %5.1f | %10.1f\n",
           v,
           order_id,
           start,
           completion);

    if (due_times[order_id] < 100.) {
      EXPECT_LE(start, due_times[order_id]) << "order" << order_id << " violates due_time";
    }
    if (earliest[order_id] > 0) {
      EXPECT_GE(start, static_cast<double>(earliest[order_id]))
        << "order" << order_id << " starts before its arrival time";
    }
  }

  printf("[arrival_nontrivial] computed_wct = %.2f\n", computed_wct);

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  printf("[arrival_nontrivial] solver WCT   = %.2f\n",
         objectives.at(objective_t::WEIGHTED_COMPLETION_TIME));
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), computed_wct, 1e-3);
}

/**
 * TEST 7 — arrival time: late order arrival flips optimal ordering
 *
 * Same 2-order setup (k_transit_times), service_times={1,1}, weights={2,1}.
 *
 * Without arrival constraints:
 *   Route A (order_0 first):  C_0=1+1=2, C_1=2+2+1=5, WCT = 2*2 + 1*5 = 9   <-- optimal
 *
 *   Wait, using service=1:
 *   Route A: C_0 = t(0→1) + s_0 = 1+1 = 2
 *            C_1 = C_0 + t(1→2) + s_1 = 2+2+1 = 5    WCT = 2*2+1*5 = 9
 *   Route B: C_1 = t(0→2) + s_1 = 3+1 = 4
 *            C_0 = C_1 + t(2→1) + s_0 = 4+2+1 = 7    WCT = 1*4+2*7 = 18
 *   → Route A is optimal (WCT=9).
 *
 * With order_0 earliest_time = 7 (order_0 physically arrives at t=7):
 *   Route A (order_0 first):
 *     start(order_0) = max(1, 7) = 7  [waits for order_0 to arrive]
 *     C_0 = 7+1 = 8
 *     start(order_1) = max(8+2, 0) = 10
 *     C_1 = 10+1 = 11    WCT = 2*8 + 1*11 = 27
 *   Route B (order_1 first):
 *     start(order_1) = max(3, 0) = 3
 *     C_1 = 3+1 = 4
 *     start(order_0) = max(4+2, 7) = max(6, 7) = 7   [order_0 arrives exactly at t=7]
 *     C_0 = 7+1 = 8    WCT = 1*4 + 2*8 = 20        <-- optimal
 *
 * The solver must pick Route B; WCT = 20.
 * order_0 arrives (via order_time_windows earliest) at t=7.
 * latest is set large so TIME-dimension upper-bound never binds.
 */
TEST(soft_time, arrival_time_flips_order)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<int> order_locations = {1, 2};
  std::vector<int> service_times   = {1, 1};
  std::vector<int> order_weights   = {2, 1};
  // order_0 arrives at t=7; order_1 arrives immediately (t=0)
  std::vector<int> earliest = {7, 0};
  std::vector<int> latest   = {10000, 10000};  // non-binding upper bound

  auto v_cost_matrix         = cuopt::device_copy(k_cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(k_transit_times, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);
  auto v_earliest            = cuopt::device_copy(earliest, stream);
  auto v_latest              = cuopt::device_copy(latest, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());
  data_model.set_order_time_windows(v_earliest.data(), v_latest.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(3);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);

  // Route B (order_1 first) is optimal: WCT = 1*4 + 2*8 = 20
  EXPECT_EQ(host_route.route[1], 1);  // order_1 first
  EXPECT_EQ(host_route.route[2], 0);  // order_0 second

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 20.0, 1e-3);
}

/**
 * TEST 8 — vehicle_order_cost steers order-to-vehicle assignment alongside WCT
 *
 * 2 vehicles, 2 orders (at locations 1 and 2).
 * Transit times: t(0->1)=1, t(0->2)=3.  Service times: {1, 1}.
 * Order weights: {1.0, 1.0}.
 *
 * vehicle_order_cost:
 *   vehicle_0: {0.0, 1000.0}  -> vehicle_0 strongly prefers order_0
 *   vehicle_1: {1000.0, 0.0}  -> vehicle_1 strongly prefers order_1
 *
 * Without vehicle_order_cost both assignments have identical WCT = 2+4 = 6,
 * so cost alone cannot distinguish them.  With vehicle_order_cost the solver
 * must pick: vehicle_0 serves order_0, vehicle_1 serves order_1.
 *
 * Expected WCT:
 *   vehicle_0: C(order_0) = t(0->1) + s_0 = 1+1 = 2
 *   vehicle_1: C(order_1) = t(0->2) + s_1 = 3+1 = 4
 *   WCT = 1*2 + 1*4 = 6
 */
TEST(soft_time, vehicle_order_cost_steers_assignment)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  const int n_locations = 3;  // depot=0, order_0=1, order_1=2
  const int n_vehicles  = 2;
  const int n_orders    = 2;

  std::vector<int> order_locations = {1, 2};
  std::vector<int> service_times   = {1, 1};
  std::vector<int> order_weights   = {1, 1};

  // vehicle_0 prefers order_0, vehicle_1 prefers order_1
  std::vector<int> costs_vehicle0 = {0, 1000};
  std::vector<int> costs_vehicle1 = {1000, 0};

  auto v_cost_matrix         = cuopt::device_copy(k_cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(k_transit_times, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);
  auto v_costs_vehicle0      = cuopt::device_copy(costs_vehicle0, stream);
  auto v_costs_vehicle1      = cuopt::device_copy(costs_vehicle1, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, n_locations, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());
  data_model.set_vehicle_order_cost(0, v_costs_vehicle0.data(), n_orders);
  data_model.set_vehicle_order_cost(1, v_costs_vehicle1.data(), n_orders);
  data_model.set_min_vehicles(n_vehicles);

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(3);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);

  // Build order->vehicle assignment map
  std::unordered_map<int, int> assignment;
  for (size_t i = 0; i < host_route.route.size(); ++i) {
    int order = host_route.route[i];
    if (order >= 0) { assignment[order] = host_route.truck_id[i]; }
  }

  // order_0 must go to vehicle_0, order_1 must go to vehicle_1
  EXPECT_EQ(assignment[0], 0);
  EXPECT_EQ(assignment[1], 1);

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 6.0, 1e-3);
}

/**
 * TEST 9 — baseline + due_time + arrival time + events (pre-scheduled breaks)
 *
 * Builds on TEST 6 (arrival_time_nontrivial_ten_orders_four_vehicles, WCT=143) by adding
 * one pre-scheduled event per vehicle: earliest=latest=10, duration=3 (busy t=10..13).
 *
 * Same 11 locations as TEST 3-6 (no extra break location — breaks can occur at any
 * existing order location; 0 extra transit for in-place breaks).
 * Same order data, due time constraints (order1,order5 due_time=4), and arrival constraints
 * (order4 earliest=4, order2 earliest=3) as TEST 6.
 *
 * Break at t=10 is chosen so due-time-constrained orders (order1 finishes ≤t=9,
 * order5 ≤t=8) can always be committed before the event fires.  Vehicles that carry
 * a 3rd order (e.g. order3 normally starting at t=8) must now absorb the break first,
 * pushing that order's completion to t≥14 → WCT rises above 143.
 *
 * Checks: SUCCESS; all orders served; due_time/arrival constraints still satisfied;
 * computed WCT matches solver WCT (consistency).
 */
TEST(soft_time, ten_orders_four_vehicles_with_events)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  const int n_locations    = 11;  // depot=0, lots=1..10 (no extra break location)
  const int n_vehicles     = 4;
  const int n_orders       = 10;
  const int break_earliest = 10;  // earliest == latest => exact event time
  const int break_latest   = 10;
  const int break_duration = 3;  // event t=10..13

  std::vector<float> transit_matrix(n_locations * n_locations);
  std::vector<float> cost_matrix(n_locations * n_locations);
  for (int i = 0; i < n_locations; ++i) {
    for (int j = 0; j < n_locations; ++j) {
      transit_matrix[i * n_locations + j] = (i == j) ? 0.f : 1.f;
      cost_matrix[i * n_locations + j]    = (i == j) ? 0.f : 0.001f;
    }
  }

  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> service_times   = {2, 5, 1, 3, 2, 4, 3, 1, 2, 3};
  std::vector<int> order_weights   = {3, 1, 4, 2, 5, 1, 3, 2, 4, 2};
  // Same due time and arrival constraints as TEST 6
  std::vector<int> due_times = {100, 4, 100, 100, 100, 4, 100, 100, 100, 100};
  std::vector<int> earliest  = {0, 0, 3, 0, 4, 0, 0, 0, 0, 0};
  std::vector<int> latest = {10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000};

  // One break per vehicle, exact time t=10, at any of the existing order locations
  std::vector<int> break_locations_h = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> v_e_h(n_vehicles, break_earliest);
  std::vector<int> v_l_h(n_vehicles, break_latest);
  std::vector<int> v_d_h(n_vehicles, break_duration);

  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(transit_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);
  auto v_due_times           = cuopt::device_copy(due_times, stream);
  auto v_earliest            = cuopt::device_copy(earliest, stream);
  auto v_latest              = cuopt::device_copy(latest, stream);
  auto v_break_locations     = cuopt::device_copy(break_locations_h, stream);
  auto v_e                   = cuopt::device_copy(v_e_h, stream);
  auto v_l                   = cuopt::device_copy(v_l_h, stream);
  auto v_d                   = cuopt::device_copy(v_d_h, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, n_locations, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());
  data_model.set_order_due_times(v_due_times.data());
  data_model.set_order_time_windows(v_earliest.data(), v_latest.data());
  data_model.set_break_locations(v_break_locations.data(), v_break_locations.size());
  data_model.add_break_dimension(v_e.data(), v_l.data(), v_d.data());
  data_model.set_min_vehicles(n_vehicles);

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(5);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  ASSERT_EQ(host_route.unserviced_nodes.size(), 0u);

  // Walk route: track per-vehicle time, handle break nodes (node_type==3)
  const int k_depot = 0, k_break = 3;
  std::vector<double> tool_time(n_vehicles, 0.0);
  double computed_wct = 0.0;

  printf("[with_events] node   | v | order | start | completion\n");
  for (int i = 0; i < static_cast<int>(host_route.route.size()); i++) {
    int ntype = host_route.node_types[i];
    int v     = host_route.truck_id[i];
    if (ntype == k_depot) {
      printf("[with_events]  depot  | v%d\n", v);
      continue;
    }
    double arrival = tool_time[v] + 1.0;
    if (ntype == k_break) {
      double start = std::max(arrival, static_cast<double>(break_earliest));
      tool_time[v] = start + break_duration;
      printf("[with_events]  break  | v%d | start=%.1f end=%.1f\n", v, start, tool_time[v]);
      continue;
    }
    int order_id      = host_route.route[i];
    double start      = std::max(arrival, static_cast<double>(earliest[order_id]));
    double completion = start + service_times[order_id];
    computed_wct += order_weights[order_id] * completion;
    tool_time[v] = completion;

    printf(
      "[with_events]  service| v%d | order%-2d | %5.1f | %10.1f\n", v, order_id, start, completion);

    if (due_times[order_id] < 100.) {
      EXPECT_LE(start, due_times[order_id]) << "order" << order_id << " violates due_time";
    }
    if (earliest[order_id] > 0) {
      EXPECT_GE(start, static_cast<double>(earliest[order_id]))
        << "order" << order_id << " starts before its arrival time";
    }
  }

  printf("[with_events] computed_wct = %.2f\n", computed_wct);

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  printf("[with_events] solver WCT   = %.2f  (baseline=122, +due_time=127, +arrival=143)\n",
         objectives.at(objective_t::WEIGHTED_COMPLETION_TIME));
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), computed_wct, 1e-3);
}

/**
 * TEST 10 — heavy transit between two orders overrides weight-only priority
 *
 * 3 orders, 1 vehicle.
 * Locations: 0=depot, 1=order_0, 2=order_1, 3=order_2
 * Service times: {2, 1, 2}
 * Order weights: {4, 1, 4}
 * Transit: uniform=1 except t(1,3)=t(3,1)=20 (heavy idle between order_0 and order_2)
 *
 * Without heavy transit (uniform=1):
 *   Route order_0→order_2→order_1: C_0=3, C_2=6, C_1=8   WCT=4*3+4*6+1*8=44  <-- would be optimal
 *
 * With t(1,3)=t(3,1)=20:
 *   Route order_0→order_2→order_1 (adjacent): C_0=3, C_2=3+20+2=25, C_1=27  WCT=139
 *   Route order_0→order_1→order_2 (separated): C_0=3, C_1=5, C_2=8          WCT=4*3+1*5+4*8=49
 *   (symmetric: order_2→order_1→order_0 also gives WCT=49)
 *
 * Expected: WCT=49; order_0 (order 0) and order_2 (order 2) are not adjacent.
 */
TEST(soft_time, idle_time_overrides_weight_priority)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  const int n_locations = 4;  // depot=0, lot_0=1, lot_1=2, lot_2=3
  const int n_vehicles  = 1;
  const int n_orders    = 3;

  // Transit: uniform 1 except t(1,3)=t(3,1)=20
  // clang-format off
  std::vector<float> transit_matrix = {
     0,  1,  1,  1,  // depot
     1,  0,  1, 20,  // order_0 -> heavy transit to order_2
     1,  1,  0,  1,  // order_1
     1, 20,  1,  0   // order_2 -> heavy transit to order_0
  };
  // clang-format on
  // Cost matrix: small uniform values so WCT dominates
  std::vector<float> cost_matrix(n_locations * n_locations, 0.001f);
  for (int i = 0; i < n_locations; ++i)
    cost_matrix[i * n_locations + i] = 0.f;

  std::vector<int> order_locations = {1, 2, 3};
  std::vector<int> service_times   = {2, 1, 2};
  std::vector<int> order_weights   = {4, 1, 4};

  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(transit_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, n_locations, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(3);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  ASSERT_EQ(host_route.unserviced_nodes.size(), 0u);

  // Extract the service-node order (skip DEPOT nodes, node_types==0)
  std::vector<int> serve_order;
  for (int i = 0; i < static_cast<int>(host_route.route.size()); i++) {
    if (host_route.node_types[i] != 0) { serve_order.push_back(host_route.route[i]); }
  }
  ASSERT_EQ(static_cast<int>(serve_order.size()), n_orders);

  // order_0 (order 0) and order_2 (order 2) must NOT be adjacent: the heavy transit
  // t(1,3)=20 makes any route with them adjacent far worse than WCT=49.
  for (int i = 0; i + 1 < static_cast<int>(serve_order.size()); i++) {
    bool adj = (serve_order[i] == 0 && serve_order[i + 1] == 2) ||
               (serve_order[i] == 2 && serve_order[i + 1] == 0);
    EXPECT_FALSE(adj) << "order_0 and order_2 are adjacent at positions " << i << " and " << (i + 1)
                      << "; order_1 should separate them to avoid 20-unit transit penalty";
  }

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  printf("[idle_time] solver WCT = %.2f  (expected 49.0)\n",
         objectives.at(objective_t::WEIGHTED_COMPLETION_TIME));
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 49.0, 1e-3);
}

/**
 * TEST 11 — pre-scheduled events modeled as exact-time vehicle breaks
 *
 * 3 orders, 1 vehicle, 1 pre-scheduled event.
 * Locations: 0=depot, 1=order_0, 2=order_1, 3=order_2  (no separate break location;
 * the event can occur at any existing order location — 0 extra transit).
 * Service times: {2, 1, 2}
 * Order weights: {3, 1, 3}
 * Transit: uniform=1 everywhere off-diagonal (4x4)
 * Event: earliest=latest=4, duration=5  (vehicle busy t=4..9)
 *
 * At most 1 order fits before the event: order_0 or order_2 (s=2) finishes at t=3,
 * vehicle waits at current location until t=4 and takes the break in place.
 *
 * Solver-optimal route: order_2(loc3) → break(loc1) → order_0(loc1) → order_1(loc2)
 *   C_2 = 1+2 = 3
 *   break at loc1: transit(loc3→loc1)=1, arrival=4, start=max(4,4)=4, end=9
 *   order_0 at loc1: transit(loc1→loc1)=0, start=9, C_0 = 9+2 = 11
 *   order_1 at loc2: transit(loc1→loc2)=1, start=12, C_1 = 12+1 = 13
 *   WCT = 3*3 + 3*11 + 1*13 = 9+33+13 = 55
 * (The solver saves 1 transit unit by placing the break at order_0's location.)
 *
 * Checks: SUCCESS; reported WCT==55; no service node starts in the break window [4,9).
 */
TEST(soft_time, events_disrupt_scheduling)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  const int n_locations    = 4;  // depot=0, lot_0=1, lot_1=2, lot_2=3
  const int n_vehicles     = 1;
  const int n_orders       = 3;
  const int break_earliest = 4;
  const int break_duration = 5;  // event from t=4 to t=9

  // Transit: uniform 1 off-diagonal (4x4)
  std::vector<float> transit_matrix(n_locations * n_locations);
  std::vector<float> cost_matrix(n_locations * n_locations);
  for (int i = 0; i < n_locations; ++i) {
    for (int j = 0; j < n_locations; ++j) {
      transit_matrix[i * n_locations + j] = (i == j) ? 0.f : 1.f;
      cost_matrix[i * n_locations + j]    = (i == j) ? 0.f : 0.001f;
    }
  }

  std::vector<int> order_locations = {1, 2, 3};
  std::vector<int> service_times   = {2, 1, 2};
  std::vector<int> order_weights   = {3, 1, 3};

  // One break per vehicle: earliest=latest=4, duration=5
  // Break can occur at any existing order location (no dedicated break node)
  std::vector<int> break_locations_h = {1, 2, 3};
  std::vector<int> v_e_h             = {break_earliest};
  std::vector<int> v_l_h             = {break_earliest};  // latest == earliest => exact start
  std::vector<int> v_d_h             = {break_duration};

  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(transit_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_order_weights       = cuopt::device_copy(order_weights, stream);
  auto v_break_locations     = cuopt::device_copy(break_locations_h, stream);
  auto v_e                   = cuopt::device_copy(v_e_h, stream);
  auto v_l                   = cuopt::device_copy(v_l_h, stream);
  auto v_d                   = cuopt::device_copy(v_d_h, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, n_locations, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_weights(v_order_weights.data());
  data_model.set_break_locations(v_break_locations.data(), v_break_locations.size());
  data_model.add_break_dimension(v_e.data(), v_l.data(), v_d.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(3);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  ASSERT_EQ(host_route.unserviced_nodes.size(), 0u);

  // Walk route to verify no lot starts during [break_earliest, break_earliest+break_duration)
  // and to independently compute WCT.
  // node_type_t values: DEPOT=0, PICKUP=1, DELIVERY=2, BREAK=3
  const int k_depot = 0, k_break = 3;
  double tool_time    = 0.0;
  double computed_wct = 0.0;

  printf("[events_disrupt] node   | v | order | start | completion\n");
  for (int i = 0; i < static_cast<int>(host_route.route.size()); i++) {
    int ntype = host_route.node_types[i];
    if (ntype == k_depot) {
      printf("[events_disrupt]  depot  | v%d\n", host_route.truck_id[i]);
      continue;
    }
    if (ntype == k_break) {
      // Break occurs at an existing lot location (no extra transit); tool waits in place
      double start = std::max(tool_time, static_cast<double>(break_earliest));
      tool_time    = start + break_duration;
      printf("[events_disrupt]  break  | v%d | start=%.1f end=%.1f\n",
             host_route.truck_id[i],
             start,
             tool_time);
      continue;
    }
    // Service node: transit=1 from previous position (uniform off-diagonal matrix)
    double arrival    = tool_time + 1.0;
    int order_id      = host_route.route[i];
    double start      = arrival;  // no lot arrival constraints in this test
    double completion = start + service_times[order_id];
    computed_wct += order_weights[order_id] * completion;
    tool_time = completion;

    printf("[events_disrupt]  service| v%d | order%-2d | %5.1f | %10.1f\n",
           host_route.truck_id[i],
           order_id,
           start,
           completion);

    // No service node should start inside the break window [4, 9)
    bool in_break_window = (start >= break_earliest && start < break_earliest + break_duration);
    EXPECT_FALSE(in_break_window) << "order" << order_id << " starts at t=" << start
                                  << " inside break window [" << break_earliest << ", "
                                  << (break_earliest + break_duration) << ")";
  }

  printf("[events_disrupt] computed_wct=%.2f\n", computed_wct);

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  printf("[events_disrupt] solver WCT  =%.2f  (expected 55.0)\n",
         objectives.at(objective_t::WEIGHTED_COMPLETION_TIME));
  // Optimal: solver places break at order_0's location (transit=0 after break),
  // which saves 1 unit vs placing break at a different location → WCT=55 not 59.
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 55.0, 1e-3);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
