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

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class lot_schedule_route_t {
 public:
  lot_schedule_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                       lot_schedule_dimension_info_t& dim_info_)
    : dim_info(dim_info_),
      lot_weight(0, sol_handle_->get_stream()),
      node_id(0, sol_handle_->get_stream()),
      fwd_completion(0, sol_handle_->get_stream()),
      fwd_wct(0, sol_handle_->get_stream()),
      bwd_weight_sum(0, sol_handle_->get_stream()),
      bwd_wct_rel(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero lot_schedule_route_t ctr");
  }

  lot_schedule_route_t(const lot_schedule_route_t& other,
                       solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(other.dim_info),
      lot_weight(other.lot_weight, sol_handle_->get_stream()),
      node_id(other.node_id, sol_handle_->get_stream()),
      fwd_completion(other.fwd_completion, sol_handle_->get_stream()),
      fwd_wct(other.fwd_wct, sol_handle_->get_stream()),
      bwd_weight_sum(other.bwd_weight_sum, sol_handle_->get_stream()),
      bwd_wct_rel(other.bwd_wct_rel, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("lot_schedule_route_t copy_ctr");
  }

  lot_schedule_route_t& operator=(lot_schedule_route_t&& other) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    lot_weight.resize(max_nodes_per_route, stream);
    node_id.resize(max_nodes_per_route, stream);
    fwd_completion.resize(max_nodes_per_route, stream);
    fwd_wct.resize(max_nodes_per_route, stream);
    bwd_weight_sum.resize(max_nodes_per_route, stream);
    bwd_wct_rel.resize(max_nodes_per_route, stream);
  }

  // -----------------------------------------------------------------------
  struct view_t {
    bool is_empty() const { return fwd_completion.empty(); }

    DI lot_schedule_node_t<i_t, f_t> get_node(i_t idx) const
    {
      lot_schedule_node_t<i_t, f_t> node;
      node.lot_weight     = lot_weight[idx];
      node.node_id        = node_id[idx];
      node.fwd_completion = fwd_completion[idx];
      node.fwd_wct        = fwd_wct[idx];
      node.bwd_weight_sum = bwd_weight_sum[idx];
      node.bwd_wct_rel    = bwd_wct_rel[idx];
      return node;
    }

    DI void set_node(i_t idx, const lot_schedule_node_t<i_t, f_t>& node)
    {
      lot_weight[idx] = node.lot_weight;
      node_id[idx]    = node.node_id;
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const lot_schedule_node_t<i_t, f_t>& node)
    {
      fwd_completion[idx] = node.fwd_completion;
      fwd_wct[idx]        = node.fwd_wct;
    }

    DI void set_backward_data(i_t idx, const lot_schedule_node_t<i_t, f_t>& node)
    {
      bwd_weight_sum[idx] = node.bwd_weight_sum;
      bwd_wct_rel[idx]    = node.bwd_wct_rel;
    }

    DI void copy_forward_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(fwd_completion.subspan(write_start), orig.fwd_completion.subspan(start_idx), size);
      block_copy(fwd_wct.subspan(write_start), orig.fwd_wct.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(bwd_weight_sum.subspan(write_start), orig.bwd_weight_sum.subspan(start_idx), size);
      block_copy(bwd_wct_rel.subspan(write_start), orig.bwd_wct_rel.subspan(start_idx), size);
    }

    DI void copy_fixed_route_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(lot_weight.subspan(write_start), orig.lot_weight.subspan(start_idx), size);
      block_copy(node_id.subspan(write_start), orig.node_id.subspan(start_idx), size);
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes_route,
                         objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      // At the return depot (position n_nodes_route), bwd_weight_sum = 0 and bwd_wct_rel = 0,
      // so total WCT = fwd_wct[n_nodes_route] which equals fwd_wct[last_service_node].
      obj_cost[objective_t::WEIGHTED_COMPLETION_TIME] = fwd_wct[n_nodes_route];
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const lot_schedule_dimension_info_t dim_info_, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info = dim_info_;
      // Layout: lot_weight | node_id | fwd_completion | fwd_wct | bwd_weight_sum | bwd_wct_rel
      // node_id is i_t, everything else is double
      i_t stride                            = n_nodes_route + 1;
      i_t* sh_ptr                           = shmem;
      thrust::tie(v.lot_weight, sh_ptr)     = wrap_ptr_as_span<double>(sh_ptr, stride);
      thrust::tie(v.fwd_completion, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, stride);
      thrust::tie(v.fwd_wct, sh_ptr)        = wrap_ptr_as_span<double>(sh_ptr, stride);
      thrust::tie(v.bwd_weight_sum, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, stride);
      thrust::tie(v.bwd_wct_rel, sh_ptr)    = wrap_ptr_as_span<double>(sh_ptr, stride);
      thrust::tie(v.node_id, sh_ptr)        = wrap_ptr_as_span<i_t>(sh_ptr, stride);
      return thrust::make_tuple(v, sh_ptr);
    }

    lot_schedule_dimension_info_t dim_info;
    raft::device_span<double> lot_weight;
    raft::device_span<i_t> node_id;
    raft::device_span<double> fwd_completion;
    raft::device_span<double> fwd_wct;
    raft::device_span<double> bwd_weight_sum;
    raft::device_span<double> bwd_wct_rel;
  };

  // -----------------------------------------------------------------------
  view_t view()
  {
    view_t v;
    v.dim_info       = dim_info;
    v.lot_weight     = raft::device_span<double>{lot_weight.data(), lot_weight.size()};
    v.node_id        = raft::device_span<i_t>{node_id.data(), node_id.size()};
    v.fwd_completion = raft::device_span<double>{fwd_completion.data(), fwd_completion.size()};
    v.fwd_wct        = raft::device_span<double>{fwd_wct.data(), fwd_wct.size()};
    v.bwd_weight_sum = raft::device_span<double>{bwd_weight_sum.data(), bwd_weight_sum.size()};
    v.bwd_wct_rel    = raft::device_span<double>{bwd_wct_rel.data(), bwd_wct_rel.size()};
    return v;
  }

  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] lot_schedule_dimension_info_t dim_info_)
  {
    // 5 double arrays + 1 i_t array, each of size route_size + 1
    return 5 * (route_size + 1) * sizeof(double) + (route_size + 1) * sizeof(i_t);
  }

  lot_schedule_dimension_info_t dim_info;

  rmm::device_uvector<double> lot_weight;
  rmm::device_uvector<i_t> node_id;
  rmm::device_uvector<double> fwd_completion;
  rmm::device_uvector<double> fwd_wct;
  rmm::device_uvector<double> bwd_weight_sum;
  rmm::device_uvector<double> bwd_wct_rel;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
