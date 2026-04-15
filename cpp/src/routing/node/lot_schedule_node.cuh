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
#include <routing/structures.hpp>

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
 * Forward state — two parallel passes:
 *
 *   Real pass (for WCT):
 *     actual_start[k]   = max(fwd_completion[k-1] + transit_{k-1,k}, earliest_time[k])
 *     fwd_completion[k] = actual_start[k] + service_time[k]   (real departure; WCT only)
 *     fwd_wct[k]        = fwd_wct[k-1] + lot_weight[k] * fwd_completion[k]
 *
 *   Shadow pass (for qtime metric, mirrors TIME dimension's HY formulation):
 *     shadow_arrival[k]       = fwd_qtime_dep[k-1] + service[k-1] + transit_{k-1,k}
 *     shadow_actual_start[k]  = max(shadow_arrival[k], earliest_time[k])
 *     fwd_qtime_excess[k]    += max(0, shadow_actual_start[k] - max_qtime[k])
 *     fwd_qtime_dep[k]        = min(shadow_actual_start[k], max_qtime[k])   ← clamped
 *
 * The shadow pass is independent of the real pass. Clamping fwd_qtime_dep prevents the
 * shadow departure from cascading unclamped delays into the next node's shadow arrival, which
 * would cause double-counting in combine/get_cost (same reason TIME clamps departure_forward).
 * The backward propagation is consistent with the shadow: it uses service + transit to
 * compute the latest allowable shadow departure (bwd_qtime_dep) at each node.
 *
 * Backward WCT state accumulates weight sums and relative WCT (assuming zero
 * waiting due to earliest_time) so that combine() is O(1) at any split point.
 * The approximation is exact when no earliest_time waits occur in the suffix.
 *
 * Qtime constraint: lot k must START processing by max_qtime[k] (relative to t=0).
 * Violation = max(0, actual_start_k - max_qtime_k). One-sided upper-bound constraint,
 * so bwd_qtime_excess is always 0 (no backward violation from a lower bound).
 *
 * Arc value for this dimension = travel_time(from, to) only.
 * Service time is fetched from vehicle_info.order_service_times at propagation time.
 *
 * Depot/break nodes must be initialised with lot_weight = 0 and node_info set to the
 * appropriate NodeInfo (DEPOT or BREAK type). Service time for breaks is read from
 * vehicle_info.break_durations[node_info.node()] at propagation time.
 */
template <typename i_t, typename f_t>
class lot_schedule_node_t {
 public:
  // ---- Fixed data (set once from problem input) ----
  double lot_weight = 0.0;    //! Importance weight of this lot; 0 for non-service nodes
  NodeInfo<i_t> node_info{};  //! Node identity: type (DEPOT/DELIVERY/BREAK) + index

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
  //! Shadow clamped departure for qtime metric: min(shadow_actual_start, max_qtime).
  //! Used by the shadow pass for transit (fwd_qtime_dep + service + transit → next shadow arrival)
  //! and as the junction point in combine/get_cost. NOT used for real transit (WCT uses
  //! fwd_completion). Analogous to TIME's departure_forward clamped at window_end.
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
  // Service-time helper: mirrors get_transit_time's logic in arc_value.hpp.
  // DEPOT → 0, BREAK → break_durations[node()], SERVICE → order_service_times[node()].
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  HDI static double get_service_time(const lot_schedule_node_t& n,
                                     const VehicleInfo<f_t, is_device>& vehicle_info) noexcept
  {
    if (n.node_info.is_depot()) { return 0.; }
    if (n.node_info.is_break()) {
      return vehicle_info.break_durations.empty()
               ? 0.
               : static_cast<double>(vehicle_info.break_durations[n.node_info.node()]);
    }
    return vehicle_info.order_service_times.empty()
             ? 0.
             : static_cast<double>(vehicle_info.order_service_times[n.node_info.node()]);
  }

  // -----------------------------------------------------------------------
  // calculate_forward
  // Called as:  this.calculate_forward(next, travel_time, vehicle_info)
  // arc = travel_time(this -> next)
  //
  // Two independent passes:
  //
  // Real pass (WCT): uses fwd_completion (real departure) for transit.
  //   actual_start = max(fwd_completion + transit, earliest_time[next])
  //   next.fwd_completion = actual_start + service[next]
  //
  // Shadow pass (qtime metric): uses fwd_qtime_dep (clamped) + service[this] for transit.
  //   shadow_arrival      = fwd_qtime_dep + service[this] + transit
  //   shadow_actual_start = max(shadow_arrival, earliest_time[next])
  //   next.fwd_qtime_excess += max(0, shadow_actual_start - max_qtime[next])
  //   next.fwd_qtime_dep    = min(shadow_actual_start, max_qtime[next])   ← clamped
  //
  // Using fwd_qtime_dep (clamped) for shadow transit prevents a late actual_start from
  // cascading unclamped into subsequent shadow arrivals and double-counting the violation.
  // The backward is consistent: it uses service + transit to propagate bwd_qtime_dep.
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  void HDI calculate_forward(lot_schedule_node_t& next,
                             double travel_time,
                             const VehicleInfo<f_t, is_device>& vehicle_info) const noexcept
  {
    double service_time_next = get_service_time<is_device>(next, vehicle_info);
    double service_time_this = get_service_time<is_device>(*this, vehicle_info);

    // --- Real pass: WCT ---
    // fwd_completion = actual_start + service_this (real departure).
    double arrival_real = fwd_completion + travel_time;
    double actual_start = (next.earliest_time > 0.) ? max(arrival_real, next.earliest_time) : arrival_real;
    next.fwd_completion = actual_start + service_time_next;
    next.fwd_wct        = fwd_wct + next.lot_weight * next.fwd_completion;

    // --- Shadow pass: qtime metric ---
    // fwd_qtime_dep = clamped shadow departure (≤ max_qtime[this]).
    double shadow_arrival = fwd_qtime_dep + service_time_this + travel_time;
    double shadow_start   = (next.earliest_time > 0.) ? max(shadow_arrival, next.earliest_time)
                                                       : shadow_arrival;
    next.fwd_qtime_excess =
      fwd_qtime_excess + (next.max_qtime > 0. ? max(0., shadow_start - next.max_qtime) : 0.);
    // Clamp shadow departure to max_qtime (mirrors TIME clamping departure_forward to window_end).
    next.fwd_qtime_dep =
      (next.max_qtime > 0. && shadow_start > next.max_qtime) ? next.max_qtime : shadow_start;
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
    double service_time_prev = get_service_time<is_device>(prev, vehicle_info);
    prev.bwd_weight_sum      = prev.lot_weight + bwd_weight_sum;
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
  // Qtime infeasibility (shadow pass — mirrors TIME dimension combine):
  //   shadow_arrival = fwd_qtime_dep[prev] + service[prev] + travel_time
  //   shadow_start   = max(shadow_arrival, earliest_time[next])
  //   excess         = fwd_excess[prev] + max(0, shadow_start - bwd_qtime_dep[next])
  //
  // Using fwd_qtime_dep (clamped) + service + transit for shadow arrival is consistent with
  // how calculate_forward and calculate_backward propagate the shadow. If fwd_completion were
  // used instead, the combine would be inconsistent with get_cost when prev has a violation.
  // bwd_qtime_excess is always 0 (one-sided upper-bound constraint).
  // -----------------------------------------------------------------------
  static HDI double combine(const lot_schedule_node_t& prev,
                            const lot_schedule_node_t& next,
                            const VehicleInfo<f_t>& vehicle_info,
                            double travel_time) noexcept
  {
    double service_time_prev = get_service_time<true>(prev, vehicle_info);
    double shadow_arrival    = prev.fwd_qtime_dep + service_time_prev + travel_time;
    double shadow_start =
      (next.earliest_time > 0.) ? max(shadow_arrival, next.earliest_time) : shadow_arrival;
    return prev.fwd_qtime_excess + max(0., shadow_start - next.bwd_qtime_dep);
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
  // Qtime: total infeasibility = fwd_qtime_excess[this] + max(0, fwd_qtime_dep - bwd_qtime_dep)
  //   fwd_qtime_excess[this] = violations accumulated in [0..this] (inclusive).
  //   fwd_qtime_dep[this]    = clamped to max_qtime, so the junction term captures the downstream
  //                            budget gap: how much too late we're starting relative to what
  //                            the suffix [this+1..n] requires (via bwd_qtime_dep).
  //   bwd_qtime_excess = 0 (one-sided upper-bound, no backward violation).
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  HDI void get_cost(const lot_schedule_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const lot_schedule_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    double service_time = get_service_time<is_device>(*this, vehicle_info);
    double handoff_time = fwd_completion - service_time;  // = actual_start[this]
    obj_cost[objective_t::WEIGHTED_COMPLETION_TIME] =
      prev_node.fwd_wct + handoff_time * bwd_weight_sum + bwd_wct_rel;

    if (dim_info.has_qtime) {
      // fwd_qtime_excess already includes violations at this node (accumulated in calculate_forward).
      // fwd_qtime_dep is clamped, so the junction term does not double-count those violations.
      inf_cost[dim_t::LOT_SCHEDULE] =
        fwd_qtime_excess + max(0., fwd_qtime_dep - bwd_qtime_dep);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
