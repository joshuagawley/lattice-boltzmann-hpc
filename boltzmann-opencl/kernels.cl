#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS 9

kernel void computation(global float *src_cells,
                        global float *dst_cells,
                        global const int *obstacles,
                        const int nx, const int ny,
                        const float omega,
                        global float *global_vels,
                        local  float *local_vels,
                        const float density, const float accel,
                        const int iteration) {

  const int grid_size = nx * ny;

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  float tot_u = 0.f;   /* accumulated magnitudes of velocity for each cell */

  const int local_id_x = get_local_id(0);
  const int local_id_y = get_local_id(1);

  const int local_size_x = get_local_size(0);
  const int local_size_y = get_local_size(1);

  const int ii = get_global_id(0);
  const int jj = get_global_id(1);
  const int offset = ii + jj * nx;

  /* determine indices of axis-direction neighbours
   * respecting periodic boundary conditions (wrap around) 
   */
  const int y_n = (jj + 1) & (ny - 1);
  const int y_s = (jj + ny - 1) & (ny - 1);
  const int x_e = (ii + 1) & (nx - 1);
  const int x_w = (ii + nx - 1) & (nx - 1);

  const float speed0 = src_cells[(0 * grid_size) + (offset)];
  const float speed1 = src_cells[(1 * grid_size) + (x_w + jj*nx)];
  const float speed2 = src_cells[(2 * grid_size) + (ii + y_s*nx)];
  const float speed3 = src_cells[(3 * grid_size) + (x_e + jj*nx)];
  const float speed4 = src_cells[(4 * grid_size) + (ii + y_n*nx)];
  const float speed5 = src_cells[(5 * grid_size) + (x_w + y_s*nx)];
  const float speed6 = src_cells[(6 * grid_size) + (x_e + y_s*nx)];
  const float speed7 = src_cells[(7 * grid_size) + (x_e + y_n*nx)];
  const float speed8 = src_cells[(8 * grid_size) + (x_w + y_n*nx)];

  /* compute local density total */
  const float local_density = speed0 + speed1 + speed2
                      + speed3 + speed4 + speed5
                      + speed6 + speed7 + speed8;

  /* compute x velocity component */
  const float u_x = (speed1 + speed5 + speed8 - (speed3 + speed6 + speed7)) / local_density;
  /* compute y velocity component */
  const float u_y = (speed2 + speed5 + speed6 - (speed4 + speed7 + speed8)) / local_density;

  const float u_sq = u_x * u_x + u_y * u_y;

  /* directional velocity components */
  const float u1 = u_x;        /* east */
  const float u2 = u_y;        /* north */
  const float u3 = -u_x;       /* west */
  const float u4 = -u_y;       /* south */
  const float u5 = u_x + u_y;  /* north-east */
  const float u6 = -u_x + u_y; /* north-west */
  const float u7 = -u_x - u_y; /* south-west */
  const float u8 = u_x - u_y;  /* south-east */

  /* equilibrium densities */
  /* zero velocity density: weight w0 */
  const float d_equ0 = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
  /* axis speeds: weight w1 */
  const float d_equ1 = w1 * local_density * (1.f + u1 / c_sq + (u1 * u1) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
  const float d_equ2 = w1 * local_density * (1.f + u2 / c_sq + (u2 * u2) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
  const float d_equ3 = w1 * local_density * (1.f + u3 / c_sq + (u3 * u3) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
  const float d_equ4 = w1 * local_density * (1.f + u4 / c_sq + (u4 * u4) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
  /* axis speeds: weight w2 */
  const float d_equ5 = w2 * local_density * (1.f + u5 / c_sq + (u5 * u5) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
  const float d_equ6 = w2 * local_density * (1.f + u6 / c_sq + (u6 * u6) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
  const float d_equ7 = w2 * local_density * (1.f + u7 / c_sq + (u7 * u7) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
  const float d_equ8 = w2 * local_density * (1.f + u8 / c_sq + (u8 * u8) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));

  /* relaxation or rebound step */
  /* if the cell contains an obstacle, we proprogate and rebound */
  /* if the cell does not contain an obstacle, we do the relaxation step */
#define RELAX_OR_REBOUND(i, d_equ, speed_if_not_obstacle, speed_if_obstacle) \
  dst_cells[(i * grid_size) + offset] = (obstacles[offset]) \
    ? speed_if_obstacle \
    : (speed_if_not_obstacle + omega * (d_equ - speed_if_not_obstacle))

  RELAX_OR_REBOUND(0, d_equ0, speed0, speed0);
  RELAX_OR_REBOUND(1, d_equ1, speed1, speed3);
  RELAX_OR_REBOUND(2, d_equ2, speed2, speed4);
  RELAX_OR_REBOUND(3, d_equ3, speed3, speed1);
  RELAX_OR_REBOUND(4, d_equ4, speed4, speed2);
  RELAX_OR_REBOUND(5, d_equ5, speed5, speed7);
  RELAX_OR_REBOUND(6, d_equ6, speed6, speed8);
  RELAX_OR_REBOUND(7, d_equ7, speed7, speed5);
  RELAX_OR_REBOUND(8, d_equ8, speed8, speed6); 

#undef RELAX_OR_REBOUND

  /* accumulate the norm of x- and y- velocity components */
  tot_u = (obstacles[offset]) ? 0 : sqrt((u_x * u_x) + (u_y * u_y));

  local_vels[local_id_x + local_id_y * local_size_x] = tot_u;

  /* ACCELERATE_FLOW */

  /* compute weighting factors */
  const float accel_w1 = density * accel / 9.f;
  const float accel_w2 = density * accel / 36.f;

  /* if the cell is not occupied and we don't send a negative density */
  const int accel_flow_condition = ((jj == ny - 2)
                          && !obstacles[offset]
                          && (dst_cells[(3 * grid_size) + offset] - accel_w1) > 0.f
                          && (dst_cells[(6 * grid_size) + offset] - accel_w2) > 0.f
                          && (dst_cells[(7 * grid_size) + offset] - accel_w2) > 0.f);

   /* increase 'east-side' densities */
  dst_cells[(1 * grid_size) + offset] += (accel_flow_condition) ? accel_w1 : 0;
  dst_cells[(5 * grid_size) + offset] += (accel_flow_condition) ? accel_w2 : 0;
  dst_cells[(8 * grid_size) + offset] += (accel_flow_condition) ? accel_w2 : 0;
  /* decrease 'west-side' densities */
  dst_cells[(3 * grid_size) + offset] -= (accel_flow_condition) ? accel_w1 : 0;
  dst_cells[(6 * grid_size) + offset] -= (accel_flow_condition) ? accel_w2 : 0;
  dst_cells[(7 * grid_size) + offset] -= (accel_flow_condition) ? accel_w2 : 0;

  const int group_id_x = get_group_id(0);
  const int group_id_y = get_group_id(1);

  const int work_groups_x = get_global_size(0) / local_size_x;
  const int work_groups_y = get_global_size(1) / local_size_y;
  const int work_groups = work_groups_x * work_groups_y;

  const int group_size = local_size_x * local_size_y;

  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if ((local_id_x + local_id_y * local_size_x) < stride) {
      local_vels[local_id_x + local_id_y * local_size_x] += local_vels[(local_id_x + local_id_y * local_size_x) + stride];
    }
  }

  /* Wait until reduction is finished before writing to host */
  barrier(CLK_LOCAL_MEM_FENCE);

  if (local_id_x == 0 && local_id_y == 0) {
    global_vels[(group_id_x + group_id_y * work_groups_x) + (iteration * work_groups)] = local_vels[0];
  }
}

kernel void reduce(global float *vels,
                   global float *av_vels,
                   const float tot_cells,
                   const int work_groups) {

  const int iter = get_global_id(0);
  const int local_id = get_local_id(0);

  float vel = 0.f;

  if (local_id == 0) {
    for (int group_id = 0; group_id < work_groups; ++group_id) {
      vel += vels[(iter * work_groups) + group_id];
    }
    av_vels[iter] = vel / tot_cells;
  }

}