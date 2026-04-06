/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * Unit tests for the LOT_SCHEDULE dimension (weighted completion time objective).
 *
 * Problem mapping: lots = orders, tools = vehicles.
 *   - lot_weight:   per-order priority weight
 *   - service_time: per-order processing time on the tool
 *   - transit time: idle time mandated between lots (travel time matrix)
 *   - WCT = sum_k( lot_weight[k] * completion_time[k] )
 *   - completion_time[k] = arrival_at_k + service_time[k]
 *
 * Test setup mirrors the hand-worked example in the code review:
 *   3 locations: 0=depot, 1=lot_0, 2=lot_1
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
//         depot  lot_0  lot_1
//  depot [  0      1      3  ]
//  lot_0 [  1      0      2  ]
//  lot_1 [  3      2      0  ]
static const std::vector<float> k_transit_times = {0, 1, 3, 1, 0, 2, 3, 2, 0};

// Uniform cost matrix so routing cost does not influence lot ordering.
static const std::vector<float> k_cost_matrix = {0, 1, 1, 1, 0, 1, 1, 1, 0};

/**
 * TEST 1 — correct ordering and WCT value
 *
 * lots:       lot_0 (w=2, s=3),  lot_1 (w=1, s=4)
 *
 * Route A: depot -> lot_0 -> lot_1 -> depot
 *   C_0 = 0 + t(0->1) + s_0 = 0 + 1 + 3 = 4
 *   C_1 = 4 + t(1->2) + s_1 = 4 + 2 + 4 = 10
 *   WCT = 2*4 + 1*10 = 18   <-- optimal
 *
 * Route B: depot -> lot_1 -> lot_0 -> depot
 *   C_1 = 0 + t(0->2) + s_1 = 0 + 3 + 4 = 7
 *   C_0 = 7 + t(2->1) + s_0 = 7 + 2 + 3 = 12
 *   WCT = 1*7 + 2*12 = 31
 *
 * Expected: solver picks Route A; WCT reported == 18.
 */
TEST(lot_scheduling, correct_order_and_wct)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<int> order_locations = {1, 2};      // lot_0 at loc 1, lot_1 at loc 2
  std::vector<int> service_times   = {3, 4};      // s_0=3, s_1=4
  std::vector<double> lot_weights  = {2.0, 1.0};  // w_0=2, w_1=1

  auto v_cost_matrix         = cuopt::device_copy(k_cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(k_transit_times, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_lot_weights         = cuopt::device_copy(lot_weights, stream);

  // 3 locations, 1 vehicle, 2 orders
  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_lot_weights(v_lot_weights.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(2);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  // Verify route order: lot_0 first, lot_1 second
  ASSERT_EQ(host_route.route[1], 0);  // position 1 = order 0 (lot_0)
  ASSERT_EQ(host_route.route[2], 1);  // position 2 = order 1 (lot_1)

  // Verify reported WCT
  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 18.0, 1e-3);
}

/**
 * TEST 2 — higher weight on the faster lot pulls it first
 *
 * Same transit times, but now lot_1 is fast (s=1) with high weight (w=3),
 * and lot_0 is slow (s=4) with low weight (w=1).
 *
 * Route A: depot -> lot_0 -> lot_1 -> depot
 *   C_0 = 0 + 1 + 4 = 5
 *   C_1 = 5 + 2 + 1 = 8
 *   WCT = 1*5 + 3*8 = 29
 *
 * Route B: depot -> lot_1 -> lot_0 -> depot
 *   C_1 = 0 + 3 + 1 = 4
 *   C_0 = 4 + 2 + 4 = 10
 *   WCT = 3*4 + 1*10 = 22   <-- optimal
 *
 * Expected: solver picks Route B; WCT reported == 22.
 */
TEST(lot_scheduling, high_weight_lot_served_first)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  std::vector<int> order_locations = {1, 2};      // lot_0 at loc 1, lot_1 at loc 2
  std::vector<int> service_times   = {4, 1};      // s_0=4, s_1=1
  std::vector<double> lot_weights  = {1.0, 3.0};  // w_0=1, w_1=3

  auto v_cost_matrix         = cuopt::device_copy(k_cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(k_transit_times, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_lot_weights         = cuopt::device_copy(lot_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(&handle, 3, 1, 2);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_lot_weights(v_lot_weights.data());

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(2);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  // Verify route order: lot_1 (high weight, fast) first, lot_0 second
  ASSERT_EQ(host_route.route[1], 1);  // position 1 = order 1 (lot_1)
  ASSERT_EQ(host_route.route[2], 0);  // position 2 = order 0 (lot_0)

  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), 22.0, 1e-3);
}

/**
 * TEST 3 — 10 lots, 4 tools, consistency check
 *
 * 11 locations: 0=depot, 1..10=lot locations (one lot per location).
 * Transit time matrix: 0 on diagonal, 1 everywhere else (uniform 1-unit transit).
 * 4 vehicles (tools), 10 orders (lots).
 *
 * Service times: {2, 5, 1, 3, 2, 4, 3, 1, 2, 3}  (lots 0..9)
 * Lot weights:   {3, 1, 4, 2, 5, 1, 3, 2, 4, 2}  (lots 0..9)
 *
 * No hand-computed optimal is needed. The test verifies:
 *   1. Solver returns SUCCESS and serves all 10 lots.
 *   2. The WCT reported by the solver equals the WCT computed by walking the
 *      flat route array independently (consistency check).
 *
 * Independent WCT walk:
 *   For each non-depot node i in the route array:
 *     order_id       = route[i]
 *     vehicle        = truck_id[i]
 *     completion[v]  = vehicle_completion[v] + transit(1) + service_times[order_id]
 *     wct           += lot_weights[order_id] * completion[v]
 *     vehicle_completion[v] = completion[v]
 *   (transit is always 1 because all off-diagonal entries in the transit matrix are 1)
 */
TEST(lot_scheduling, ten_lots_four_tools_consistency)
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

  // lot i is at location i+1
  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> service_times   = {2, 5, 1, 3, 2, 4, 3, 1, 2, 3};
  std::vector<double> lot_weights  = {3., 1., 4., 2., 5., 1., 3., 2., 4., 2.};

  auto v_cost_matrix         = cuopt::device_copy(cost_matrix, stream);
  auto v_transit_time_matrix = cuopt::device_copy(transit_matrix, stream);
  auto v_order_locations     = cuopt::device_copy(order_locations, stream);
  auto v_service_times       = cuopt::device_copy(service_times, stream);
  auto v_lot_weights         = cuopt::device_copy(lot_weights, stream);

  cuopt::routing::data_model_view_t<int, float> data_model(
    &handle, n_locations, n_vehicles, n_orders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.add_transit_time_matrix(v_transit_time_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_order_service_times(v_service_times.data());
  data_model.set_order_lot_weights(v_lot_weights.data());
  // Force solver to use all vehicles (cuOpt minimizes vehicle count by default,
  // but lot scheduling benefits from parallel execution across all tools).
  data_model.set_min_vehicles(n_vehicles);

  cuopt::routing::solver_settings_t<int, float> settings;
  settings.set_time_limit(5);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  // host_route.print();

  // All lots must be served
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
    double contrib    = lot_weights[order_id] * completion;
    computed_wct += contrib;
    vehicle_completion[v] = completion;
    printf(
      "[route]  service   | v%-4d | lot%-2d | %10.1f | %10.1f\n", v, order_id, completion, contrib);
  }

  printf("[ten_lots_four_tools] computed_wct = %.2f\n", computed_wct);

  // Reported WCT must match independently computed WCT
  const auto& objectives = routing_solution.get_objectives();
  ASSERT_TRUE(objectives.count(objective_t::WEIGHTED_COMPLETION_TIME) > 0);
  printf("[ten_lots_four_tools] solver WCT   = %.2f\n",
         objectives.at(objective_t::WEIGHTED_COMPLETION_TIME));
  EXPECT_NEAR(objectives.at(objective_t::WEIGHTED_COMPLETION_TIME), computed_wct, 1e-3);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
