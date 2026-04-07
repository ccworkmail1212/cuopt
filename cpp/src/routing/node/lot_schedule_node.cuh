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
 * Tracks weighted completion time (WCT) for lot scheduling, and optionally
 * qtime (queue-time) infeasibility for lots that must begin processing within
 * a deadline measured from t=0.
 *
 * WCT = sum_k ( lot_weight[k] * completion_time[k] )
 * where completion_time[k] = time when the tool finishes processing lot k.
 *
 * Forward state:
 *   actual_start[k] = max(arrival_at_k, earliest_time[k])
 *   fwd_completion[k] = actual_start[k] + service_time[k]
 *   fwd_wct[k]        = fwd_wct[k-1] + lot_weight[k] * fwd_completion[k]
 *   fwd_qtime_dep[k]  = actual_start[k]   (tracks actual start for qtime checking)
 *
 * Backward WCT state accumulates weight sums and relative WCT (assuming zero
 * waiting due to earliest_time) so that combine() is O(1) at any split point.
 * The approximation is exact when no earliest_time waits occur in the suffix.
 *
 * Qtime constraint: lot k must START processing by max_qtime[k] (relative to t=0).
 * Violation = max(0, actual_start_k - max_qtime_k). Modelled using warping
 * identical to TIME dimension time-window warping, but one-sided (no lower-bound
 * excess), so bwd_qtime_excess is always 0.
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

  //! Earliest allowable start time (lot arrival time). 0 = no constraint.
  double earliest_time = 0.0;
  //! Qtime deadline: lot must start processing by this time (t=0 reference). 0 = no constraint.
  double max_qtime = 0.0;

  // ---- Forward WCT state ----
  //! Completion time of this lot = actual_start + service_time
  double fwd_completion = 0.0;
  //! Accumulated WCT: sum_{i=0}^{k} lot_weight[i] * completion[i]
  double fwd_wct = 0.0;

  // ---- Backward WCT state ----
  //! Sum of lot_weights from this position to end of route
  double bwd_weight_sum = 0.0;
  //! Relative WCT of suffix [k..n] assuming handoff time = 0 and no earliest_time waits
  double bwd_wct_rel = 0.0;

  // ---- Forward qtime state ----
  //! Actual START time at this lot: max(arrival, earliest_time)
  double fwd_qtime_dep = 0.0;
  //! Cumulative qtime violations up to and including this lot
  double fwd_qtime_excess = 0.0;

  // ---- Backward qtime state ----
  //! Latest allowable START of this lot so that this lot and all subsequent lots satisfy qtime.
  //! Propagated as: bwd_qtime_dep[k] = min(max_qtime[k], bwd_qtime_dep[k+1] - s_k -
  //! transit_{k,k+1})
  double bwd_qtime_dep = 0.0;
  //! Always 0: qtime is a one-sided upper-bound constraint, no backward excess.
  double bwd_qtime_excess = 0.0;

  // -----------------------------------------------------------------------
  // calculate_forward
  // Called as:  this.calculate_forward(next, travel_time, vehicle_info)
  // arc = travel_time(this -> next)
  // Service time of `next` is fetched from vehicle_info.
  //
  // actual_start_next = max(fwd_qtime_dep[this] + service[this] + travel, earliest_time[next])
  // fwd_completion[next] = actual_start_next + service_time[next]
  // fwd_wct[next]        = fwd_wct[this] + lot_weight[next] * fwd_completion[next]
  // fwd_qtime_dep[next]  = actual_start_next
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  void HDI calculate_forward(lot_schedule_node_t& next,
                             double travel_time,
                             const VehicleInfo<f_t, is_device>& vehicle_info) const noexcept
  {
    double service_time_next =
      (next.node_id >= 0 && !vehicle_info.order_service_times.empty())
        ? static_cast<double>(vehicle_info.order_service_times[next.node_id])
        : 0.;
    double service_time_this = (node_id >= 0 && !vehicle_info.order_service_times.empty())
                                 ? static_cast<double>(vehicle_info.order_service_times[node_id])
                                 : 0.;

    double arrival      = fwd_qtime_dep + service_time_this + travel_time;
    double actual_start = (next.earliest_time > 0.) ? max(arrival, next.earliest_time) : arrival;

    // --- WCT forward ---
    next.fwd_completion = actual_start + service_time_next;
    next.fwd_wct        = fwd_wct + next.lot_weight * next.fwd_completion;

    // --- Qtime forward ---
    next.fwd_qtime_dep = actual_start;
    next.fwd_qtime_excess =
      fwd_qtime_excess + (next.max_qtime > 0. ? max(0., actual_start - next.max_qtime) : 0.);
  }

  // -----------------------------------------------------------------------
  // calculate_backward
  // Called as:  this.calculate_backward(prev, travel_time, vehicle_info)
  // arc = travel_time(prev -> this)
  // Service time of `prev` is fetched from vehicle_info.
  //
  // WCT backward recurrence (relative to a hypothetical handoff time of 0,
  // ignoring earliest_time waits — same approximation as TIME dimension):
  //   prev.bwd_weight_sum = prev.lot_weight + this.bwd_weight_sum
  //   prev.bwd_wct_rel    = prev.bwd_weight_sum * service_time[prev]
  //                       + this.bwd_wct_rel
  //                       + this.bwd_weight_sum * travel_time
  //
  // Qtime backward recurrence:
  //   limit = this.bwd_qtime_dep - service_time[prev] - travel_time
  //   prev.bwd_qtime_dep = (prev.max_qtime > 0) ? min(prev.max_qtime, limit) : limit
  //   prev.bwd_qtime_excess = 0 (one-sided constraint)
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  void HDI calculate_backward(lot_schedule_node_t& prev,
                              double travel_time,
                              const VehicleInfo<f_t, is_device>& vehicle_info) const noexcept
  {
    // --- WCT backward ---
    double service_time_prev =
      (prev.node_id >= 0 && !vehicle_info.order_service_times.empty())
        ? static_cast<double>(vehicle_info.order_service_times[prev.node_id])
        : 0.;
    prev.bwd_weight_sum = prev.lot_weight + bwd_weight_sum;
    prev.bwd_wct_rel =
      prev.bwd_weight_sum * service_time_prev + bwd_wct_rel + bwd_weight_sum * travel_time;

    // --- Qtime backward ---
    // latest start of prev = latest start of this - service[prev] - transit(prev->this)
    double limit          = bwd_qtime_dep - service_time_prev - travel_time;
    prev.bwd_qtime_dep    = (prev.max_qtime > 0.) ? min(prev.max_qtime, limit) : limit;
    prev.bwd_qtime_excess = 0.;
  }

  // -----------------------------------------------------------------------
  // combine
  // Returns infeasibility delta at the join point (prev, next) with arc travel_time.
  // WCT is a pure objective — no infeasibility contribution.
  // Qtime infeasibility:
  //   arrival        = fwd_qtime_dep[prev] + service_time[prev] + travel_time
  //   actual_start   = max(arrival, earliest_time[next])
  //   excess         = fwd_excess[prev] + max(0, actual_start - bwd_dep[next])
  // bwd_qtime_excess is always 0 (one-sided constraint).
  // -----------------------------------------------------------------------
  static HDI double combine(const lot_schedule_node_t& prev,
                            const lot_schedule_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            double travel_time) noexcept
  {
    double service_time_prev =
      (prev.node_id >= 0 && !vehicle_info.order_service_times.empty())
        ? static_cast<double>(vehicle_info.order_service_times[prev.node_id])
        : 0.;
    double arrival      = prev.fwd_qtime_dep + service_time_prev + travel_time;
    double actual_start = (next.earliest_time > 0.) ? max(arrival, next.earliest_time) : arrival;
    return prev.fwd_qtime_excess + max(0., actual_start - next.bwd_qtime_dep);
  }

  // -----------------------------------------------------------------------
  // forward/backward excess
  // -----------------------------------------------------------------------
  HDI double forward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return fwd_qtime_excess;
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return bwd_qtime_excess;  // always 0
  }

  HDI bool forward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            double weight       = 1.0,
                            double excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                             double weight       = 1.0,
                             double excess_limit = 0.) const noexcept
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  // -----------------------------------------------------------------------
  // get_cost
  // Called with prev_node = node[k-1] (fwd already propagated into `this`).
  //
  // WCT: the combine invariant gives total WCT at split point k-1 → k:
  //   handoff_time = this.fwd_completion - service_time[this]  (= actual_start[this])
  //   total_wct   = prev.fwd_wct
  //               + handoff_time * this.bwd_weight_sum
  //               + this.bwd_wct_rel
  //
  // Qtime: infeasibility = fwd_excess + 0 + max(0, fwd_dep - bwd_dep)
  //   (analogous to TIME dimension: excess_f + excess_b + max(0, dep_f - dep_b))
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  HDI void get_cost(const lot_schedule_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const lot_schedule_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    double service_time = (node_id >= 0 && !vehicle_info.order_service_times.empty())
                            ? static_cast<double>(vehicle_info.order_service_times[node_id])
                            : 0.;
    double handoff_time = fwd_completion - service_time;  // = actual_start[this]
    obj_cost[objective_t::WEIGHTED_COMPLETION_TIME] =
      prev_node.fwd_wct + handoff_time * bwd_weight_sum + bwd_wct_rel;

    if (dim_info.has_qtime) {
      // Use prev_node.fwd_qtime_excess (violations [0..k-1]) to avoid double-counting:
      // the junction term max(0, fwd_dep[k] - bwd_dep[k]) already accounts for violations at k+.
      inf_cost[dim_t::LOT_SCHEDULE] =
        prev_node.fwd_qtime_excess + max(0., fwd_qtime_dep - bwd_qtime_dep);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
