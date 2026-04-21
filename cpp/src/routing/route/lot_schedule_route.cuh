/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/lot_schedule_node.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"
#include "routing/structures.hpp"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

#include <cstdint>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class lot_schedule_route_t {
 public:
  // K = entries per route position in the backward qtime arrays
  static constexpr i_t K = static_cast<i_t>(MAX_LOT_SCHED_ROUTE_SIZE);

  lot_schedule_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                       lot_schedule_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      lot_weight(0, sol_handle_->get_stream()),
      node_info(0, sol_handle_->get_stream()),
      earliest_time(0, sol_handle_->get_stream()),
      max_qtime(0, sol_handle_->get_stream()),
      fwd_completion(0, sol_handle_->get_stream()),
      fwd_wct(0, sol_handle_->get_stream()),
      bwd_weight_sum(0, sol_handle_->get_stream()),
      bwd_wct_rel(0, sol_handle_->get_stream()),
      fwd_qtime_obj(0, sol_handle_->get_stream()),
      bwd_qtime_b(0, sol_handle_->get_stream()),
      bwd_qtime_w(0, sol_handle_->get_stream()),
      bwd_n_constrained(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero lot_schedule_route_t ctr");
  }

  lot_schedule_route_t(const lot_schedule_route_t& other,
                       solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(other.dim_info),
      lot_weight(other.lot_weight, sol_handle_->get_stream()),
      node_info(other.node_info, sol_handle_->get_stream()),
      earliest_time(other.earliest_time, sol_handle_->get_stream()),
      max_qtime(other.max_qtime, sol_handle_->get_stream()),
      fwd_completion(other.fwd_completion, sol_handle_->get_stream()),
      fwd_wct(other.fwd_wct, sol_handle_->get_stream()),
      bwd_weight_sum(other.bwd_weight_sum, sol_handle_->get_stream()),
      bwd_wct_rel(other.bwd_wct_rel, sol_handle_->get_stream()),
      fwd_qtime_obj(other.fwd_qtime_obj, sol_handle_->get_stream()),
      bwd_qtime_b(other.bwd_qtime_b, sol_handle_->get_stream()),
      bwd_qtime_w(other.bwd_qtime_w, sol_handle_->get_stream()),
      bwd_n_constrained(other.bwd_n_constrained, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("lot_schedule_route_t copy_ctr");
  }

  lot_schedule_route_t& operator=(lot_schedule_route_t&& other) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    lot_weight.resize(max_nodes_per_route, stream);
    node_info.resize(max_nodes_per_route, stream);
    earliest_time.resize(max_nodes_per_route, stream);
    max_qtime.resize(max_nodes_per_route, stream);
    fwd_completion.resize(max_nodes_per_route, stream);
    fwd_wct.resize(max_nodes_per_route, stream);
    bwd_weight_sum.resize(max_nodes_per_route, stream);
    bwd_wct_rel.resize(max_nodes_per_route, stream);
    fwd_qtime_obj.resize(max_nodes_per_route, stream);
    bwd_qtime_b.resize(max_nodes_per_route * K, stream);
    bwd_qtime_w.resize(max_nodes_per_route * K, stream);
    bwd_n_constrained.resize(max_nodes_per_route, stream);
  }

  // -----------------------------------------------------------------------
  struct view_t {
    bool is_empty() const { return fwd_completion.empty(); }

    DI lot_schedule_node_t<i_t, f_t> get_node(i_t idx) const
    {
      lot_schedule_node_t<i_t, f_t> node;
      node.lot_weight        = lot_weight[idx];
      node.node_info         = node_info[idx];
      node.earliest_time     = earliest_time[idx];
      node.max_qtime         = max_qtime[idx];
      node.fwd_completion    = fwd_completion[idx];
      node.fwd_wct           = fwd_wct[idx];
      node.bwd_weight_sum    = bwd_weight_sum[idx];
      node.bwd_wct_rel       = bwd_wct_rel[idx];
      node.fwd_qtime_obj     = fwd_qtime_obj[idx];
      node.bwd_n_constrained = bwd_n_constrained[idx];
      for (i_t j = 0; j < K; j++) {
        node.bwd_qtime_b[j] = bwd_qtime_b[idx * K + j];
        node.bwd_qtime_w[j] = bwd_qtime_w[idx * K + j];
      }
      return node;
    }

    DI void set_node(i_t idx, const lot_schedule_node_t<i_t, f_t>& node)
    {
      lot_weight[idx]    = node.lot_weight;
      node_info[idx]     = node.node_info;
      earliest_time[idx] = node.earliest_time;
      max_qtime[idx]     = node.max_qtime;
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const lot_schedule_node_t<i_t, f_t>& node)
    {
      fwd_completion[idx] = node.fwd_completion;
      fwd_wct[idx]        = node.fwd_wct;
      fwd_qtime_obj[idx]  = node.fwd_qtime_obj;
    }

    DI void set_backward_data(i_t idx, const lot_schedule_node_t<i_t, f_t>& node)
    {
      bwd_weight_sum[idx]    = node.bwd_weight_sum;
      bwd_wct_rel[idx]       = node.bwd_wct_rel;
      bwd_n_constrained[idx] = node.bwd_n_constrained;
      for (i_t j = 0; j < K; j++) {
        bwd_qtime_b[idx * K + j] = node.bwd_qtime_b[j];
        bwd_qtime_w[idx * K + j] = node.bwd_qtime_w[j];
      }
    }

    DI void copy_forward_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(fwd_completion.subspan(write_start), orig.fwd_completion.subspan(start_idx), size);
      block_copy(fwd_wct.subspan(write_start), orig.fwd_wct.subspan(start_idx), size);
      block_copy(fwd_qtime_obj.subspan(write_start), orig.fwd_qtime_obj.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(bwd_weight_sum.subspan(write_start), orig.bwd_weight_sum.subspan(start_idx), size);
      block_copy(bwd_wct_rel.subspan(write_start), orig.bwd_wct_rel.subspan(start_idx), size);
      block_copy(
        bwd_n_constrained.subspan(write_start), orig.bwd_n_constrained.subspan(start_idx), size);
      block_copy(
        bwd_qtime_b.subspan(write_start * K), orig.bwd_qtime_b.subspan(start_idx * K), size * K);
      block_copy(
        bwd_qtime_w.subspan(write_start * K), orig.bwd_qtime_w.subspan(start_idx * K), size * K);
    }

    DI void copy_fixed_route_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(lot_weight.subspan(write_start), orig.lot_weight.subspan(start_idx), size);
      block_copy(node_info.subspan(write_start), orig.node_info.subspan(start_idx), size);
      block_copy(earliest_time.subspan(write_start), orig.earliest_time.subspan(start_idx), size);
      block_copy(max_qtime.subspan(write_start), orig.max_qtime.subspan(start_idx), size);
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         [[maybe_unused]] infeasible_cost_t& inf_cost) const noexcept
    {
      // At the return depot bwd_weight_sum=0 and bwd_wct_rel=0, so total WCT = fwd_wct[n].
      obj_cost[objective_t::WEIGHTED_COMPLETION_TIME] = static_cast<double>(fwd_wct[n_nodes_route]);
      // fwd_qtime_obj[n] = exact total qtime penalty accumulated over all lots.
      if (dim_info.has_qtime) {
        obj_cost[objective_t::LOT_QTIME_PENALTY] =
          static_cast<double>(fwd_qtime_obj[n_nodes_route]);
      }
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const lot_schedule_dimension_info_t dim_info_, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info = dim_info_;
      // All fields are int32_t (4-byte aligned):
      //   fwd_wct | bwd_wct_rel | fwd_qtime_obj | lot_weight | earliest_time
      //   | max_qtime | fwd_completion | bwd_weight_sum
      //   | bwd_qtime_b (stride*K) | bwd_qtime_w (stride*K)
      //   | bwd_n_constrained (int) | node_info (NodeInfo<i_t>)
      i_t stride                               = n_nodes_route + 1;
      i_t* sh_ptr                              = shmem;
      thrust::tie(v.fwd_wct, sh_ptr)           = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.bwd_wct_rel, sh_ptr)       = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.fwd_qtime_obj, sh_ptr)     = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.lot_weight, sh_ptr)        = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.earliest_time, sh_ptr)     = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.max_qtime, sh_ptr)         = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.fwd_completion, sh_ptr)    = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.bwd_weight_sum, sh_ptr)    = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.bwd_qtime_b, sh_ptr)       = wrap_ptr_as_span<int32_t>(sh_ptr, stride * K);
      thrust::tie(v.bwd_qtime_w, sh_ptr)       = wrap_ptr_as_span<int32_t>(sh_ptr, stride * K);
      thrust::tie(v.bwd_n_constrained, sh_ptr) = wrap_ptr_as_span<int>(sh_ptr, stride);
      thrust::tie(v.node_info, sh_ptr)         = wrap_ptr_as_span<NodeInfo<i_t>>(sh_ptr, stride);
      return thrust::make_tuple(v, sh_ptr);
    }

    lot_schedule_dimension_info_t dim_info;
    raft::device_span<int32_t> lot_weight;
    raft::device_span<NodeInfo<i_t>> node_info;
    raft::device_span<int32_t> earliest_time;
    raft::device_span<int32_t> max_qtime;
    raft::device_span<int32_t> fwd_completion;
    raft::device_span<int32_t> fwd_wct;
    raft::device_span<int32_t> bwd_weight_sum;
    raft::device_span<int32_t> bwd_wct_rel;
    raft::device_span<int32_t> fwd_qtime_obj;
    raft::device_span<int32_t> bwd_qtime_b;  // size: stride * K
    raft::device_span<int32_t> bwd_qtime_w;  // size: stride * K
    raft::device_span<int> bwd_n_constrained;
  };

  // -----------------------------------------------------------------------
  view_t view()
  {
    view_t v;
    v.dim_info       = dim_info;
    v.lot_weight     = raft::device_span<int32_t>{lot_weight.data(), lot_weight.size()};
    v.node_info      = raft::device_span<NodeInfo<i_t>>{node_info.data(), node_info.size()};
    v.earliest_time  = raft::device_span<int32_t>{earliest_time.data(), earliest_time.size()};
    v.max_qtime      = raft::device_span<int32_t>{max_qtime.data(), max_qtime.size()};
    v.fwd_completion = raft::device_span<int32_t>{fwd_completion.data(), fwd_completion.size()};
    v.fwd_wct        = raft::device_span<int32_t>{fwd_wct.data(), fwd_wct.size()};
    v.bwd_weight_sum = raft::device_span<int32_t>{bwd_weight_sum.data(), bwd_weight_sum.size()};
    v.bwd_wct_rel    = raft::device_span<int32_t>{bwd_wct_rel.data(), bwd_wct_rel.size()};
    v.fwd_qtime_obj  = raft::device_span<int32_t>{fwd_qtime_obj.data(), fwd_qtime_obj.size()};
    v.bwd_qtime_b    = raft::device_span<int32_t>{bwd_qtime_b.data(), bwd_qtime_b.size()};
    v.bwd_qtime_w    = raft::device_span<int32_t>{bwd_qtime_w.data(), bwd_qtime_w.size()};
    v.bwd_n_constrained =
      raft::device_span<int>{bwd_n_constrained.data(), bwd_n_constrained.size()};
    return v;
  }

  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] lot_schedule_dimension_info_t dim_info_)
  {
    // 8 int32_t scalar arrays + 2 int32_t arrays of size K per position + int array + NodeInfo
    // array
    i_t stride = route_size + 1;
    return static_cast<size_t>(8 * stride) * sizeof(int32_t) +
           static_cast<size_t>(2 * stride * K) * sizeof(int32_t) +
           raft::alignTo(static_cast<size_t>(stride) * sizeof(int), sizeof(int32_t)) +
           raft::alignTo(static_cast<size_t>(stride) * sizeof(NodeInfo<i_t>), sizeof(int32_t));
  }

  lot_schedule_dimension_info_t dim_info;

  rmm::device_uvector<int32_t> lot_weight;
  rmm::device_uvector<NodeInfo<i_t>> node_info;
  rmm::device_uvector<int32_t> earliest_time;
  rmm::device_uvector<int32_t> max_qtime;
  rmm::device_uvector<int32_t> fwd_completion;
  rmm::device_uvector<int32_t> fwd_wct;
  rmm::device_uvector<int32_t> bwd_weight_sum;
  rmm::device_uvector<int32_t> bwd_wct_rel;
  rmm::device_uvector<int32_t> fwd_qtime_obj;
  rmm::device_uvector<int32_t> bwd_qtime_b;    // size: (route_size+1) * K
  rmm::device_uvector<int32_t> bwd_qtime_w;    // size: (route_size+1) * K
  rmm::device_uvector<int> bwd_n_constrained;  // size: route_size+1
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
