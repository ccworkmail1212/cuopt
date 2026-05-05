/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/soft_time_node.cuh"
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
class soft_time_route_t {
 public:
  // K = entries per route position in the backward lateness arrays
  static constexpr i_t K = static_cast<i_t>(MAX_SOFT_TIME_ROUTE_SIZE);

  soft_time_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                    soft_time_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      order_weight(0, sol_handle_->get_stream()),
      node_info(0, sol_handle_->get_stream()),
      earliest_time(0, sol_handle_->get_stream()),
      due_time(0, sol_handle_->get_stream()),
      fwd_completion(0, sol_handle_->get_stream()),
      fwd_wct(0, sol_handle_->get_stream()),
      bwd_weight_sum(0, sol_handle_->get_stream()),
      bwd_wct_rel(0, sol_handle_->get_stream()),
      fwd_lateness(0, sol_handle_->get_stream()),
      bwd_lateness_b(0, sol_handle_->get_stream()),
      bwd_lateness_w(0, sol_handle_->get_stream()),
      bwd_lateness_n(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero soft_time_route_t ctr");
  }

  soft_time_route_t(const soft_time_route_t& other, solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(other.dim_info),
      order_weight(other.order_weight, sol_handle_->get_stream()),
      node_info(other.node_info, sol_handle_->get_stream()),
      earliest_time(other.earliest_time, sol_handle_->get_stream()),
      due_time(other.due_time, sol_handle_->get_stream()),
      fwd_completion(other.fwd_completion, sol_handle_->get_stream()),
      fwd_wct(other.fwd_wct, sol_handle_->get_stream()),
      bwd_weight_sum(other.bwd_weight_sum, sol_handle_->get_stream()),
      bwd_wct_rel(other.bwd_wct_rel, sol_handle_->get_stream()),
      fwd_lateness(other.fwd_lateness, sol_handle_->get_stream()),
      bwd_lateness_b(other.bwd_lateness_b, sol_handle_->get_stream()),
      bwd_lateness_w(other.bwd_lateness_w, sol_handle_->get_stream()),
      bwd_lateness_n(other.bwd_lateness_n, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("soft_time_route_t copy_ctr");
  }

  soft_time_route_t& operator=(soft_time_route_t&& other) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    order_weight.resize(max_nodes_per_route, stream);
    node_info.resize(max_nodes_per_route, stream);
    earliest_time.resize(max_nodes_per_route, stream);
    due_time.resize(max_nodes_per_route, stream);
    fwd_completion.resize(max_nodes_per_route, stream);
    fwd_wct.resize(max_nodes_per_route, stream);
    bwd_weight_sum.resize(max_nodes_per_route, stream);
    bwd_wct_rel.resize(max_nodes_per_route, stream);
    fwd_lateness.resize(max_nodes_per_route, stream);
    bwd_lateness_b.resize(max_nodes_per_route * K, stream);
    bwd_lateness_w.resize(max_nodes_per_route * K, stream);
    bwd_lateness_n.resize(max_nodes_per_route, stream);
  }

  // -----------------------------------------------------------------------
  struct view_t {
    bool is_empty() const { return fwd_completion.empty(); }

    DI soft_time_node_t<i_t, f_t> get_node(i_t idx) const
    {
      soft_time_node_t<i_t, f_t> node;
      node.order_weight   = order_weight[idx];
      node.node_info      = node_info[idx];
      node.earliest_time  = earliest_time[idx];
      node.due_time       = due_time[idx];
      node.fwd_completion = fwd_completion[idx];
      node.fwd_wct        = fwd_wct[idx];
      node.bwd_weight_sum = bwd_weight_sum[idx];
      node.bwd_wct_rel    = bwd_wct_rel[idx];
      node.fwd_lateness   = fwd_lateness[idx];
      node.bwd_lateness_n = bwd_lateness_n[idx];
      for (i_t j = 0; j < K; j++) {
        node.bwd_lateness_b[j] = bwd_lateness_b[idx * K + j];
        node.bwd_lateness_w[j] = bwd_lateness_w[idx * K + j];
      }
      return node;
    }

    DI void set_node(i_t idx, const soft_time_node_t<i_t, f_t>& node)
    {
      order_weight[idx]  = node.order_weight;
      node_info[idx]     = node.node_info;
      earliest_time[idx] = node.earliest_time;
      due_time[idx]      = node.due_time;
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const soft_time_node_t<i_t, f_t>& node)
    {
      fwd_completion[idx] = node.fwd_completion;
      fwd_wct[idx]        = node.fwd_wct;
      fwd_lateness[idx]   = node.fwd_lateness;
    }

    DI void set_backward_data(i_t idx, const soft_time_node_t<i_t, f_t>& node)
    {
      bwd_weight_sum[idx] = node.bwd_weight_sum;
      bwd_wct_rel[idx]    = node.bwd_wct_rel;
      bwd_lateness_n[idx] = node.bwd_lateness_n;
      for (i_t j = 0; j < K; j++) {
        bwd_lateness_b[idx * K + j] = node.bwd_lateness_b[j];
        bwd_lateness_w[idx * K + j] = node.bwd_lateness_w[j];
      }
    }

    DI void copy_forward_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(fwd_completion.subspan(write_start), orig.fwd_completion.subspan(start_idx), size);
      block_copy(fwd_wct.subspan(write_start), orig.fwd_wct.subspan(start_idx), size);
      block_copy(fwd_lateness.subspan(write_start), orig.fwd_lateness.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(bwd_weight_sum.subspan(write_start), orig.bwd_weight_sum.subspan(start_idx), size);
      block_copy(bwd_wct_rel.subspan(write_start), orig.bwd_wct_rel.subspan(start_idx), size);
      block_copy(bwd_lateness_n.subspan(write_start), orig.bwd_lateness_n.subspan(start_idx), size);
      block_copy(bwd_lateness_b.subspan(write_start * K),
                 orig.bwd_lateness_b.subspan(start_idx * K),
                 size * K);
      block_copy(bwd_lateness_w.subspan(write_start * K),
                 orig.bwd_lateness_w.subspan(start_idx * K),
                 size * K);
    }

    DI void copy_fixed_route_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(order_weight.subspan(write_start), orig.order_weight.subspan(start_idx), size);
      block_copy(node_info.subspan(write_start), orig.node_info.subspan(start_idx), size);
      block_copy(earliest_time.subspan(write_start), orig.earliest_time.subspan(start_idx), size);
      block_copy(due_time.subspan(write_start), orig.due_time.subspan(start_idx), size);
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         [[maybe_unused]] infeasible_cost_t& inf_cost) const noexcept
    {
      // At the return depot bwd_weight_sum=0 and bwd_wct_rel=0, so total WCT = fwd_wct[n].
      obj_cost[objective_t::WEIGHTED_COMPLETION_TIME] = static_cast<double>(fwd_wct[n_nodes_route]);
      // fwd_lateness[n] = exact total lateness penalty accumulated over all orders.
      if (dim_info.has_lateness) {
        obj_cost[objective_t::LATENESS] = static_cast<double>(fwd_lateness[n_nodes_route]);
      }
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const soft_time_dimension_info_t dim_info_, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info = dim_info_;
      // All fields are int32_t (4-byte aligned):
      //   fwd_wct | bwd_wct_rel | fwd_lateness | order_weight | earliest_time
      //   | due_time | fwd_completion | bwd_weight_sum
      //   | bwd_lateness_b (stride*K) | bwd_lateness_w (stride*K)
      //   | bwd_lateness_n (int) | node_info (NodeInfo<i_t>)
      i_t stride                            = n_nodes_route + 1;
      i_t* sh_ptr                           = shmem;
      thrust::tie(v.fwd_wct, sh_ptr)        = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.bwd_wct_rel, sh_ptr)    = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.fwd_lateness, sh_ptr)   = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.order_weight, sh_ptr)   = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.earliest_time, sh_ptr)  = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.due_time, sh_ptr)       = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.fwd_completion, sh_ptr) = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.bwd_weight_sum, sh_ptr) = wrap_ptr_as_span<int32_t>(sh_ptr, stride);
      thrust::tie(v.bwd_lateness_b, sh_ptr) = wrap_ptr_as_span<int32_t>(sh_ptr, stride * K);
      thrust::tie(v.bwd_lateness_w, sh_ptr) = wrap_ptr_as_span<int32_t>(sh_ptr, stride * K);
      thrust::tie(v.bwd_lateness_n, sh_ptr) = wrap_ptr_as_span<int>(sh_ptr, stride);
      thrust::tie(v.node_info, sh_ptr)      = wrap_ptr_as_span<NodeInfo<i_t>>(sh_ptr, stride);
      return thrust::make_tuple(v, sh_ptr);
    }

    soft_time_dimension_info_t dim_info;
    raft::device_span<int32_t> order_weight;
    raft::device_span<NodeInfo<i_t>> node_info;
    raft::device_span<int32_t> earliest_time;
    raft::device_span<int32_t> due_time;
    raft::device_span<int32_t> fwd_completion;
    raft::device_span<int32_t> fwd_wct;
    raft::device_span<int32_t> bwd_weight_sum;
    raft::device_span<int32_t> bwd_wct_rel;
    raft::device_span<int32_t> fwd_lateness;
    raft::device_span<int32_t> bwd_lateness_b;  // size: stride * K
    raft::device_span<int32_t> bwd_lateness_w;  // size: stride * K
    raft::device_span<int> bwd_lateness_n;
  };

  // -----------------------------------------------------------------------
  view_t view()
  {
    view_t v;
    v.dim_info       = dim_info;
    v.order_weight   = raft::device_span<int32_t>{order_weight.data(), order_weight.size()};
    v.node_info      = raft::device_span<NodeInfo<i_t>>{node_info.data(), node_info.size()};
    v.earliest_time  = raft::device_span<int32_t>{earliest_time.data(), earliest_time.size()};
    v.due_time       = raft::device_span<int32_t>{due_time.data(), due_time.size()};
    v.fwd_completion = raft::device_span<int32_t>{fwd_completion.data(), fwd_completion.size()};
    v.fwd_wct        = raft::device_span<int32_t>{fwd_wct.data(), fwd_wct.size()};
    v.bwd_weight_sum = raft::device_span<int32_t>{bwd_weight_sum.data(), bwd_weight_sum.size()};
    v.bwd_wct_rel    = raft::device_span<int32_t>{bwd_wct_rel.data(), bwd_wct_rel.size()};
    v.fwd_lateness   = raft::device_span<int32_t>{fwd_lateness.data(), fwd_lateness.size()};
    v.bwd_lateness_b = raft::device_span<int32_t>{bwd_lateness_b.data(), bwd_lateness_b.size()};
    v.bwd_lateness_w = raft::device_span<int32_t>{bwd_lateness_w.data(), bwd_lateness_w.size()};
    v.bwd_lateness_n = raft::device_span<int>{bwd_lateness_n.data(), bwd_lateness_n.size()};
    return v;
  }

  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] soft_time_dimension_info_t dim_info_)
  {
    // 8 int32_t scalar arrays + 2 int32_t arrays of size K per position + int array + NodeInfo
    // array
    i_t stride = route_size + 1;
    return static_cast<size_t>(8 * stride) * sizeof(int32_t) +
           static_cast<size_t>(2 * stride * K) * sizeof(int32_t) +
           raft::alignTo(static_cast<size_t>(stride) * sizeof(int), sizeof(int32_t)) +
           raft::alignTo(static_cast<size_t>(stride) * sizeof(NodeInfo<i_t>), sizeof(int32_t));
  }

  soft_time_dimension_info_t dim_info;

  rmm::device_uvector<int32_t> order_weight;
  rmm::device_uvector<NodeInfo<i_t>> node_info;
  rmm::device_uvector<int32_t> earliest_time;
  rmm::device_uvector<int32_t> due_time;
  rmm::device_uvector<int32_t> fwd_completion;
  rmm::device_uvector<int32_t> fwd_wct;
  rmm::device_uvector<int32_t> bwd_weight_sum;
  rmm::device_uvector<int32_t> bwd_wct_rel;
  rmm::device_uvector<int32_t> fwd_lateness;
  rmm::device_uvector<int32_t> bwd_lateness_b;  // size: (route_size+1) * K
  rmm::device_uvector<int32_t> bwd_lateness_w;  // size: (route_size+1) * K
  rmm::device_uvector<int> bwd_lateness_n;      // size: route_size+1
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
