/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../routing_helpers.cuh"

#include <routing/fleet_info.hpp>
#include <routing/routing_details.hpp>

namespace cuopt {
namespace routing {
namespace detail {

/**
 * @brief Per-node state for the LOT_SCHEDULE dimension.
 *
 * Tracks weighted completion time (WCT) for lot scheduling.
 * WCT = sum_k ( lot_weight[k] * completion_time[k] )
 * where completion_time[k] = time when the tool finishes processing lot k.
 *
 * Forward state propagates completion times and accumulates WCT.
 * Backward state accumulates weight sums and relative WCT so that
 * combine() is O(1) at any split point.
 *
 * Arc value for this dimension = travel_time(from, to) only.
 * Service time is fetched from vehicle_info.order_service_times at propagation time.
 *
 * Depot nodes must be initialised with lot_weight = 0, node_id = -1.
 */
template <typename i_t, typename f_t>
class lot_schedule_node_t {
 public:
  // ---- Fixed data (set once from problem input) ----
  double lot_weight = 0.0;  //! Importance weight of this lot; 0 for depot nodes
  i_t node_id       = -1;   //! Order index into vehicle_info.order_service_times; -1 for depots

  // ---- Forward state ----
  //! Completion time of this lot = time tool finishes processing it
  double fwd_completion = 0.0;
  //! Accumulated WCT: sum_{i=0}^{k} lot_weight[i] * completion[i]
  double fwd_wct = 0.0;

  // ---- Backward state ----
  //! Sum of lot_weights from this position to end of route
  double bwd_weight_sum = 0.0;
  //! Relative WCT of suffix [k..n] assuming handoff time = 0
  double bwd_wct_rel = 0.0;

  // -----------------------------------------------------------------------
  // calculate_forward
  // Called as:  this.calculate_forward(next, travel_time, vehicle_info)
  // arc = travel_time(this -> next)
  // Service time of `next` is fetched from vehicle_info.
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  void HDI calculate_forward(lot_schedule_node_t& next,
                             double travel_time,
                             const VehicleInfo<f_t, is_device>& vehicle_info) const noexcept
  {
    double service_time = (next.node_id >= 0 && !vehicle_info.order_service_times.empty())
                            ? static_cast<double>(vehicle_info.order_service_times[next.node_id])
                            : 0.;
    next.fwd_completion = fwd_completion + travel_time + service_time;
    next.fwd_wct        = fwd_wct + next.lot_weight * next.fwd_completion;
  }

  // -----------------------------------------------------------------------
  // calculate_backward
  // Called as:  this.calculate_backward(prev, travel_time, vehicle_info)
  // arc = travel_time(prev -> this)
  // Service time of `prev` is fetched from vehicle_info.
  //
  // Backward recurrence (relative to a hypothetical handoff time of 0):
  //   prev.bwd_weight_sum = prev.lot_weight + this.bwd_weight_sum
  //   prev.bwd_wct_rel    = prev.bwd_weight_sum * service_time[prev]
  //                       + this.bwd_wct_rel
  //                       + this.bwd_weight_sum * travel_time
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  void HDI calculate_backward(lot_schedule_node_t& prev,
                              double travel_time,
                              const VehicleInfo<f_t, is_device>& vehicle_info) const noexcept
  {
    double service_time_prev =
      (prev.node_id >= 0 && !vehicle_info.order_service_times.empty())
        ? static_cast<double>(vehicle_info.order_service_times[prev.node_id])
        : 0.;
    prev.bwd_weight_sum = prev.lot_weight + bwd_weight_sum;
    prev.bwd_wct_rel =
      prev.bwd_weight_sum * service_time_prev + bwd_wct_rel + bwd_weight_sum * travel_time;
  }

  // -----------------------------------------------------------------------
  // combine
  // Pure objective — no infeasibility contribution.
  // -----------------------------------------------------------------------
  static HDI double combine([[maybe_unused]] const lot_schedule_node_t& prev,
                            [[maybe_unused]] const lot_schedule_node_t& next,
                            [[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] double travel_time) noexcept
  {
    return 0.;
  }

  // -----------------------------------------------------------------------
  // forward/backward excess — always 0 (no hard constraint in this dimension)
  // -----------------------------------------------------------------------
  HDI double forward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return 0.;
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return 0.;
  }

  HDI bool forward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] double weight       = 1.0,
                            [[maybe_unused]] double excess_limit = 0.) const noexcept
  {
    return true;
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                             [[maybe_unused]] double weight       = 1.0,
                             [[maybe_unused]] double excess_limit = 0.) const noexcept
  {
    return true;
  }

  // -----------------------------------------------------------------------
  // get_cost
  // Called with prev_node = node[k-1] (fwd already propagated into `this`).
  //
  // The combine invariant gives total WCT at split point k-1 → k:
  //   handoff_time = prev.fwd_completion + travel(k-1, k)
  //               = this.fwd_completion - service_time[this]
  //   total_wct   = prev.fwd_wct
  //               + handoff_time * this.bwd_weight_sum
  //               + this.bwd_wct_rel
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  HDI void get_cost(const lot_schedule_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    [[maybe_unused]] const lot_schedule_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    [[maybe_unused]] infeasible_cost_t& inf_cost) const noexcept
  {
    double service_time = (node_id >= 0 && !vehicle_info.order_service_times.empty())
                            ? static_cast<double>(vehicle_info.order_service_times[node_id])
                            : 0.;
    double handoff_time = fwd_completion - service_time;
    obj_cost[objective_t::WEIGHTED_COMPLETION_TIME] =
      prev_node.fwd_wct + handoff_time * bwd_weight_sum + bwd_wct_rel;
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
