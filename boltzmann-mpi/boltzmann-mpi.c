/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"

#define ROOT 0

/* struct to hold the parameter values */
typedef struct {
  int size;
  int rank;
  int left;
  int right;
  int start_y;
  int end_y;
  int nrows;

  int nx;              /* no. of cells in x-direction */
  int ny;              /* no. of cells in y-direction */
  int max_iters;       /* no. of iterations */
  int reynolds_dim;    /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int unblocked_cells; /* total unblocked cells */
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
  float *speed0;
  float *speed1;
  float *speed2;
  float *speed3;
  float *speed4;
  float *speed5;
  float *speed6;
  float *speed7;
  float *speed8;
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle
 * densities */
int initialise(const char *paramfile, const char *obstaclefile, t_param *params,
               t_speed **all_cells_ptr, t_speed **local_src_cells_ptr,
               t_speed **local_dst_cells_ptr, int **local_obstacles_ptr,
               int **all_obstacles_ptr, float **local_vels_ptr);

void initialise_speed(t_speed *speed, const int size);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed *src_cells, t_speed *dst_cells,
               const int *obstacles);
void halo_transfer(const t_param params, t_speed *cells, const int send_idx,
                   const int recv_idx, const int dst, const int src);
int write_values(const t_param params, const t_speed *cells,
                 const int *obstacles, const float *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(t_speed **all_cells_ptr, t_speed **local_src_cells_ptr,
             t_speed **local_dst_cells_ptr, int **all_obstacles_ptr,
             int **local_obstacles_ptr, float **local_vels_ptr);

void finalise_speed(t_speed *speed);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed *cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed *cells, const int *obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed *cells, const int *obstacles);

/* utility functions */
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char *argv[]) {
  /* Set-up MPI distribution environment */
  MPI_Init(&argc, &argv);

  char *paramfile = NULL;          /* name of the input parameter file */
  char *obstaclefile = NULL;       /* name of a the input obstacle file */
  t_param params;                  /* struct to hold parameter values */
  t_speed *all_cells = NULL;       /* grid containing final fluid densities */
  t_speed *local_src_cells = NULL; /* grid containing fluid densities */
  t_speed *local_dst_cells = NULL; /* scratch space */
  int *local_obstacles = NULL;     /* grid indicating which cells are blocked */
  int *all_obstacles = NULL;
  float *local_vels =
      NULL; /* a record of the total velocity computed for each timestep */
  struct timeval timstr; /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic,
      col_toc; /* floating point numbers to calculate elapsed wallclock time */

  MPI_Comm_size(MPI_COMM_WORLD, &params.size);
  MPI_Comm_rank(MPI_COMM_WORLD, &params.rank);

  /* parse the command line */
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values
   * from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic = tot_tic;
  initialise(paramfile, obstaclefile, &params, &all_cells, &local_src_cells,
             &local_dst_cells, &local_obstacles, &all_obstacles, &local_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic = init_toc;

  for (int tt = 0; tt < params.max_iters; tt += 2) {
    local_vels[tt] =
        timestep(params, local_src_cells, local_dst_cells, local_obstacles);
    local_vels[tt + 1] =
        timestep(params, local_dst_cells, local_src_cells, local_obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("velocity: %.12E\n", vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;

  /* Collate data from ranks here */

  /* Collect the grid from the other ranks */
  const int num_cells = params.nx * params.nrows;
  int recv_counts[params.size];
  int displs[params.size];
  MPI_Gather(&num_cells, 1, MPI_INT, &recv_counts, 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  if (params.rank == ROOT) {
    displs[0] = 0;
    for (int i = 1; i < params.size; ++i)
      displs[i] = displs[i - 1] + recv_counts[i - 1];
  }

#define GATHER_CELLS(local_speed, speed)                                       \
  MPI_Gatherv((local_speed) + params.nx, num_cells, MPI_FLOAT, (speed),        \
              (const int *)&recv_counts, (const int *)&displs, MPI_FLOAT, 0,   \
              MPI_COMM_WORLD)

  GATHER_CELLS(local_src_cells->speed0, all_cells->speed0);
  GATHER_CELLS(local_src_cells->speed1, all_cells->speed1);
  GATHER_CELLS(local_src_cells->speed2, all_cells->speed2);
  GATHER_CELLS(local_src_cells->speed3, all_cells->speed3);
  GATHER_CELLS(local_src_cells->speed4, all_cells->speed4);
  GATHER_CELLS(local_src_cells->speed5, all_cells->speed5);
  GATHER_CELLS(local_src_cells->speed6, all_cells->speed6);
  GATHER_CELLS(local_src_cells->speed7, all_cells->speed7);
  GATHER_CELLS(local_src_cells->speed8, all_cells->speed8);

#undef GATHER_CELLS

  /* Collect average velocites */
  float av_vels[params.max_iters];
  MPI_Reduce(local_vels, &av_vels, params.max_iters, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (params.rank == ROOT) {
    for (int i = 0; i < params.max_iters; ++i)
      av_vels[i] /= (float)params.unblocked_cells;
  }

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  if (params.rank == ROOT) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n",
           calc_reynolds(params, all_cells, all_obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
    write_values(params, all_cells, all_obstacles, av_vels);
    finalise(&all_cells, &local_src_cells, &local_dst_cells, &all_obstacles,
             &local_obstacles, &local_vels);
  }

  /* Close down MPI */
  MPI_Finalize();

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed *src_cells, t_speed *dst_cells,
               const int *obstacles) {
  float tot_u = 0;

  /* compute weighting factors */
  const float accel_w1 = params.density * params.accel / 9.f;
  const float accel_w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int second_row = params.ny - 2;

  /* ACCELERATE FLOW */
  /* only accelerate flow if this rank contains the second row */
  if ((params.start_y <= second_row) && (second_row <= params.end_y)) {
    const int jj = second_row - params.start_y + 1;
#pragma omp simd
    for (int ii = 0; ii < params.nx; ++ii) {
      const int offset = ii + jj * params.nx;
      /* if the cell is not occupied and
      ** we don't send a negative density */
      const int condition = !obstacles[offset] &&
                            (src_cells->speed3[offset] - accel_w1) > 0.f &&
                            (src_cells->speed6[offset] - accel_w2) > 0.f &&
                            (src_cells->speed7[offset] - accel_w2) > 0.f;

      /* increase 'east-side' densities */
      src_cells->speed1[offset] += (condition) ? accel_w1 : 0;
      src_cells->speed5[offset] += (condition) ? accel_w2 : 0;
      src_cells->speed8[offset] += (condition) ? accel_w2 : 0;
      /* decrease 'west-side' densities */
      src_cells->speed3[offset] -= (condition) ? accel_w1 : 0;
      src_cells->speed6[offset] -= (condition) ? accel_w2 : 0;
      src_cells->speed7[offset] -= (condition) ? accel_w2 : 0;
    }
  }

  /* HALO TRANSFER */
  /* Send to the right, receive from the left */
  halo_transfer(params, src_cells, params.nrows * params.nx, 0, params.right,
                params.left);
  /* Send to the left, receive from the right */
  halo_transfer(params, src_cells, params.nx, (params.nrows + 1) * params.nx,
                params.left, params.right);

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  /* COMBINED PROPROGATE AND REBOUND
   * loop over the src_cells in the grid
   *
   * Don't iterate over halo regions;
   * these will be overwritten on next iteration anyway
   */
  for (int jj = 1; jj < params.nrows + 1; ++jj) {
    const int y_n = jj + 1;
    const int y_s = jj - 1;
#pragma omp simd
    for (int ii = 0; ii < params.nx; ++ii) {
      const int offset = ii + jj * params.nx;

      /* determine indices of axis-direction neighbours
       ** respecting periodic boundary conditions (wrap around) */
      const int x_e = (ii + 1) % params.nx;
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);

      const float speed0 = src_cells->speed0[ii + jj * params.nx];
      const float speed1 = src_cells->speed1[x_w + jj * params.nx];
      const float speed2 = src_cells->speed2[ii + y_s * params.nx];
      const float speed3 = src_cells->speed3[x_e + jj * params.nx];
      const float speed4 = src_cells->speed4[ii + y_n * params.nx];
      const float speed5 = src_cells->speed5[x_w + y_s * params.nx];
      const float speed6 = src_cells->speed6[x_e + y_s * params.nx];
      const float speed7 = src_cells->speed7[x_e + y_n * params.nx];
      const float speed8 = src_cells->speed8[x_w + y_n * params.nx];

      const float local_density = speed0 + speed1 + speed2 + speed3 + speed4 +
                                  speed5 + speed6 + speed7 + speed8;

      /* compute x velocity component */
      const float u_x =
          (speed1 + speed5 + speed8 - (speed3 + speed6 + speed7)) /
          local_density;
      /* compute y velocity component */
      const float u_y =
          (speed2 + speed5 + speed6 - (speed4 + speed7 + speed8)) /
          local_density;

      /* velocity squared */
      const float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] = u_x;        /* east */
      u[2] = u_y;        /* north */
      u[3] = -u_x;       /* west */
      u[4] = -u_y;       /* south */
      u[5] = u_x + u_y;  /* north-east */
      u[6] = -u_x + u_y; /* north-west */
      u[7] = -u_x - u_y; /* south-west */
      u[8] = u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      d_equ[0] = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
      /* axis speeds: weight w1 */
      for (int i = 1; i < 5; ++i) {
        d_equ[i] = w1 * local_density *
                   (1.f + u[i] / c_sq + (u[i] * u[i]) / (2.f * c_sq * c_sq) -
                    u_sq / (2.f * c_sq));
      }
      /* axis speeds: weight w2 */
      for (int i = 5; i < NSPEEDS; ++i) {
        d_equ[i] = w2 * local_density *
                   (1.f + u[i] / c_sq + (u[i] * u[i]) / (2.f * c_sq * c_sq) -
                    u_sq / (2.f * c_sq));
      }
      /* relaxation or rebound step */
      /* if the cell contains an obstacle, we proprogate and rebound */
      /* if the cell does not contain an obstacle, we do the relaxation step */
#define RELAX_OR_REBOUND(i, src_speed_if_not_obstacle, src_speed_if_obstacle,  \
                         dst_speed)                                            \
  dst_speed =                                                                  \
      (obstacles[offset])                                                      \
          ? (src_speed_if_obstacle)                                            \
          : ((src_speed_if_not_obstacle) +                                     \
             params.omega * (d_equ[(i)] - (src_speed_if_not_obstacle)))

      RELAX_OR_REBOUND(0, speed0, speed0, dst_cells->speed0[offset]);
      RELAX_OR_REBOUND(1, speed1, speed3, dst_cells->speed1[offset]);
      RELAX_OR_REBOUND(2, speed2, speed4, dst_cells->speed2[offset]);
      RELAX_OR_REBOUND(3, speed3, speed1, dst_cells->speed3[offset]);
      RELAX_OR_REBOUND(4, speed4, speed2, dst_cells->speed4[offset]);
      RELAX_OR_REBOUND(5, speed5, speed7, dst_cells->speed5[offset]);
      RELAX_OR_REBOUND(6, speed6, speed8, dst_cells->speed6[offset]);
      RELAX_OR_REBOUND(7, speed7, speed5, dst_cells->speed7[offset]);
      RELAX_OR_REBOUND(8, speed8, speed6, dst_cells->speed8[offset]);

      /* accumulate the norm of x- and y- velocity components */
      tot_u += (obstacles[offset]) ? 0 : sqrtf((u_x * u_x) + (u_y * u_y));
    }
  }
  /* Don't divide by total unblocked cells now,
   * we do this later when we collate all the average velocities.
   */
  return tot_u;
}

void halo_transfer(const t_param params, t_speed *cells, const int send_idx,
                   const int recv_idx, const int dst, const int src) {
  MPI_Status status;

#define SENDRECV_SPEED(speed)                                                  \
  MPI_Sendrecv((speed) + send_idx, params.nx, MPI_FLOAT, dst, params.rank,     \
               (speed) + recv_idx, params.nx, MPI_FLOAT, src, MPI_ANY_TAG,     \
               MPI_COMM_WORLD, &status)

  SENDRECV_SPEED(cells->speed0);
  SENDRECV_SPEED(cells->speed1);
  SENDRECV_SPEED(cells->speed2);
  SENDRECV_SPEED(cells->speed3);
  SENDRECV_SPEED(cells->speed4);
  SENDRECV_SPEED(cells->speed5);
  SENDRECV_SPEED(cells->speed6);
  SENDRECV_SPEED(cells->speed7);
  SENDRECV_SPEED(cells->speed8);

#undef SENDRECV_SPEED
}

float av_velocity(const t_param params, t_speed *cells, const int *obstacles) {
  float tot_u = 0.0f; /* accumulated magnitudes of velocity for each cell */

  /* loop over all non-blocked src_cells */
  for (int jj = 0; jj < params.ny; ++jj) {
#pragma omp simd
    for (int ii = 0; ii < params.nx; ++ii) {
      /* ignore occupied src_cells */
      const int offset = ii + jj * params.nx;

      /* local density total */
      const float local_density =
          cells->speed0[offset] + cells->speed1[offset] +
          cells->speed2[offset] + cells->speed3[offset] +
          cells->speed4[offset] + cells->speed5[offset] +
          cells->speed6[offset] + cells->speed7[offset] + cells->speed8[offset];

      /* x-component of velocity */
      const float u_x = (cells->speed1[offset] + cells->speed5[offset] +
                         cells->speed8[offset] -
                         (cells->speed3[offset] + cells->speed6[offset] +
                          cells->speed7[offset])) /
                        local_density;
      /* compute y velocity component */
      const float u_y = (cells->speed2[offset] + cells->speed5[offset] +
                         cells->speed6[offset] -
                         (cells->speed4[offset] + cells->speed7[offset] +
                          cells->speed8[offset])) /
                        local_density;
      /* accumulate the norm of x- and y- velocity components */
      tot_u += (obstacles[offset]) ? 0 : sqrtf((u_x * u_x) + (u_y * u_y));
    }
  }

  return tot_u / (float)params.unblocked_cells;
}

int initialise(const char *paramfile, const char *obstaclefile, t_param *params,
               t_speed **all_cells_ptr, t_speed **local_src_cells_ptr,
               t_speed **local_dst_cells_ptr, int **local_obstacles_ptr,
               int **all_obstacles_ptr, float **local_vels_ptr) {
  char message[1024]; /* message buffer */
  FILE *fp;           /* file pointer */
  int xx, yy;         /* generic array indices */
  int blocked;        /* indicates whether a cell is blocked by an obstacle */
  int retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1)
    die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1)
    die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->max_iters));

  if (retval != 1)
    die("could not read param file: max_iters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1)
    die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1)
    die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1)
    die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1)
    die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  *all_cells_ptr = (t_speed *)malloc(sizeof(t_speed));
  *local_src_cells_ptr = (t_speed *)malloc(sizeof(t_speed));
  *local_dst_cells_ptr = (t_speed *)malloc(sizeof(t_speed));

  /* Work out above/below rank ids */
  params->left = (params->size + params->rank - 1) % params->size;
  params->right = (params->rank + 1) % params->size;

  /* Calculate start and end columns for the rank */
  const int base_height = params->ny / params->size;
  const int remainder = params->ny % params->size;
  params->nrows = base_height + ((params->rank < remainder) ? 1 : 0);
  params->start_y = base_height * params->rank +
                    ((params->rank <= remainder) ? params->rank : remainder);
  params->end_y = params->start_y + params->nrows - 1;

  if (params->rank == 0) {
    /* grid for collation on master rank */
    initialise_speed(*all_cells_ptr, params->nx * params->ny);

    /* the final map of obstacles */
    *all_obstacles_ptr =
        aligned_alloc(32, sizeof(int) * (params->ny * params->nx));

    if (*all_obstacles_ptr == NULL)
      die("cannot allocate column memory for obstacles", __LINE__, __FILE__);
  }

  /* main grid */
  initialise_speed(*local_src_cells_ptr, params->nx * (params->nrows + 2));

  if ((*local_src_cells_ptr)->speed0 == NULL ||
      (*local_src_cells_ptr)->speed1 == NULL ||
      (*local_src_cells_ptr)->speed2 == NULL ||
      (*local_src_cells_ptr)->speed2 == NULL ||
      (*local_src_cells_ptr)->speed4 == NULL ||
      (*local_src_cells_ptr)->speed5 == NULL ||
      (*local_src_cells_ptr)->speed6 == NULL ||
      (*local_src_cells_ptr)->speed7 == NULL ||
      (*local_src_cells_ptr)->speed8 == NULL)
    die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  initialise_speed(*local_dst_cells_ptr, params->nx * (params->nrows + 2));

  if ((*local_dst_cells_ptr)->speed0 == NULL ||
      (*local_dst_cells_ptr)->speed1 == NULL ||
      (*local_dst_cells_ptr)->speed2 == NULL ||
      (*local_dst_cells_ptr)->speed2 == NULL ||
      (*local_dst_cells_ptr)->speed4 == NULL ||
      (*local_dst_cells_ptr)->speed5 == NULL ||
      (*local_dst_cells_ptr)->speed6 == NULL ||
      (*local_dst_cells_ptr)->speed7 == NULL ||
      (*local_dst_cells_ptr)->speed8 == NULL)
    die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles
   * strictly speaking we don't need to allocate the halo regions
   * but doing so means we can just keep track of one offset in the
   * timestep function
   */
  *local_obstacles_ptr = (int *)aligned_alloc(
      32, sizeof(int) * ((params->nrows + 2) * params->nx));

  if (*local_obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density / 9.f;
  const float w2 = params->density / 36.f;

  for (int jj = 0; jj < (params->nrows + 2); ++jj) {
#pragma omp simd
    for (int ii = 0; ii < params->nx; ++ii) {
      const int offset = ii + jj * params->nx;
      /* centre */
      (*local_src_cells_ptr)->speed0[offset] = w0;
      /* axis directions */
      (*local_src_cells_ptr)->speed1[offset] = w1;
      (*local_src_cells_ptr)->speed2[offset] = w1;
      (*local_src_cells_ptr)->speed3[offset] = w1;
      (*local_src_cells_ptr)->speed4[offset] = w1;
      /* diagonals */
      (*local_src_cells_ptr)->speed5[offset] = w2;
      (*local_src_cells_ptr)->speed6[offset] = w2;
      (*local_src_cells_ptr)->speed7[offset] = w2;
      (*local_src_cells_ptr)->speed8[offset] = w2;

      (*local_obstacles_ptr)[offset] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  const int above_y = (params->ny + params->start_y - 1) % params->ny;
  const int below_y = (params->end_y + 1) % params->ny;
  params->unblocked_cells = params->nx * params->ny;
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if (retval != 3)
      die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1)
      die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1)
      die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1)
      die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    /* Interior cells */
    if (params->start_y <= yy && yy <= params->end_y)
      (*local_obstacles_ptr)[xx + (yy - params->start_y + 1) * params->nx] =
          blocked;

    /* Halo region cells */
    if (yy == above_y)
      (*local_obstacles_ptr)[xx] = blocked;
    if (yy == below_y)
      (*local_obstacles_ptr)[xx + (params->nrows + 1) * params->nx] = blocked;

    if (params->rank == 0) {
      (*all_obstacles_ptr)[xx + yy * params->nx] = blocked;
      params->unblocked_cells -= (blocked) ? 1 : 0;
    }
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *local_vels_ptr =
      (float *)aligned_alloc(32, sizeof(float) * params->max_iters);

  return EXIT_SUCCESS;
}

inline void initialise_speed(t_speed *speed, const int size) {
  /* we align each array */
  speed->speed0 = (float *)aligned_alloc(32, sizeof(float) * size);
  speed->speed1 = (float *)aligned_alloc(32, sizeof(float) * size);
  speed->speed2 = (float *)aligned_alloc(32, sizeof(float) * size);
  speed->speed3 = (float *)aligned_alloc(32, sizeof(float) * size);
  speed->speed4 = (float *)aligned_alloc(32, sizeof(float) * size);
  speed->speed5 = (float *)aligned_alloc(32, sizeof(float) * size);
  speed->speed6 = (float *)aligned_alloc(32, sizeof(float) * size);
  speed->speed7 = (float *)aligned_alloc(32, sizeof(float) * size);
  speed->speed8 = (float *)aligned_alloc(32, sizeof(float) * size);
}

int finalise(t_speed **all_cells_ptr, t_speed **local_src_cells_ptr,
             t_speed **local_dst_cells_ptr, int **all_obstacles_ptr,
             int **local_obstacles_ptr, float **local_vels_ptr) {
  /*
  ** free up allocated memory
  */
  finalise_speed(*all_cells_ptr);
  finalise_speed(*local_src_cells_ptr);
  finalise_speed(*local_dst_cells_ptr);

  free(*all_cells_ptr);
  *all_cells_ptr = NULL;

  free(*local_src_cells_ptr);
  *local_src_cells_ptr = NULL;

  free(*local_dst_cells_ptr);
  *local_dst_cells_ptr = NULL;

  free(*local_obstacles_ptr);
  *local_obstacles_ptr = NULL;

  free(*all_obstacles_ptr);
  *all_obstacles_ptr = NULL;

  free(*local_vels_ptr);
  *local_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

inline void finalise_speed(t_speed *speed) {
  free(speed->speed0);
  free(speed->speed1);
  free(speed->speed2);
  free(speed->speed3);
  free(speed->speed4);
  free(speed->speed5);
  free(speed->speed6);
  free(speed->speed7);
  free(speed->speed8);
}

float calc_reynolds(const t_param params, t_speed *cells,
                    const int *obstacles) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim /
         viscosity;
}

float total_density(const t_param params, t_speed *cells) {
  float total = 0.f;
  for (int jj = 0; jj < params.ny; ++jj) {
#pragma omp simd
    for (int ii = 0; ii < params.nx; ++ii) {
      const int offset = ii + jj * params.nx;
      total += cells->speed0[offset];

      total += cells->speed1[offset] + cells->speed2[offset] +
               cells->speed3[offset] + cells->speed4[offset];

      total += cells->speed5[offset] + cells->speed6[offset] +
               cells->speed7[offset] + cells->speed8[offset];
    }
  }

  return total;
}

int write_values(const t_param params, const t_speed *cells,
                 const int *obstacles, const float *av_vels) {
  FILE *fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;          /* per grid cell sum of densities */
  float pressure;               /* fluid pressure in grid cell */
  float u_x;                    /* x-component of velocity in grid cell */
  float u_y;                    /* y-component of velocity in grid cell */
  float u; /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; ++jj) {
    for (int ii = 0; ii < params.nx; ++ii) {
      const int offset = ii + jj * params.nx;
      /* an occupied cell */
      if (obstacles[offset]) {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else {
        local_density = cells->speed0[offset] + cells->speed1[offset] +
                        cells->speed2[offset] + cells->speed3[offset] +
                        cells->speed4[offset] + cells->speed5[offset] +
                        cells->speed6[offset] + cells->speed7[offset] +
                        cells->speed8[offset];

        /* x-component of velocity */
        float u_x = (cells->speed1[offset] + cells->speed5[offset] +
                     cells->speed8[offset] -
                     (cells->speed3[offset] + cells->speed6[offset] +
                      cells->speed7[offset])) /
                    local_density;
        /* compute y velocity component */
        float u_y = (cells->speed2[offset] + cells->speed5[offset] +
                     cells->speed6[offset] -
                     (cells->speed4[offset] + cells->speed7[offset] +
                      cells->speed8[offset])) /
                    local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u,
              pressure, obstacles[offset]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.max_iters; ++ii) {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char *message, const int line, const char *file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char *exe) {
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
