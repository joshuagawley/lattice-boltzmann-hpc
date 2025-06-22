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
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"

/* struct to hold the parameter values */
typedef struct {
  int nx;           /* no. of src_cells in x-direction */
  int ny;           /* no. of src_cells in y-direction */
  int max_iters;    /* no. of iterations */
  int reynolds_dim; /* dimension for Reynolds number */
  float density;    /* density per link */
  float accel;      /* density redistribution */
  float omega;      /* relaxation parameter */
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
int initialise(const char *param_file, const char *obstacle_file,
               const t_param *params, t_speed **src_cells_ptr,
               t_speed **dst_cells_ptr, bool **obstacles_ptr,
               float **av_vels_ptr);

void initialise_speed(const t_param *params, t_speed *speed);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed *src_cells, t_speed *dst_cells,
               bool *obstacles);
int write_values(const t_param params, const t_speed *src_cells,
                 const bool *obstacles, const float *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, t_speed **src_cells_ptr,
             t_speed **dst_cells_ptr, bool **obstacles_ptr,
             float **av_vels_ptr);

void finalise_speed(t_speed *speed);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed *src_cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed *src_cells, bool *obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed *src_cells, bool *obstacles);

/* utility functions */
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char *argv[]) {
  char *param_file = NULL;    /* name of the input parameter file */
  char *obstacle_file = NULL; /* name of a the input obstacle file */
  t_param params;             /* struct to hold parameter values */
  t_speed *src_cells = NULL;  /* grid containing fluid densities */
  t_speed *dst_cells = NULL;  /* scratch space */
  bool *obstacles = NULL;     /* grid indicating which src_cells are blocked */
  float *av_vels =
      NULL; /* a record of the av. velocity computed for each timestep */
  struct timeval timstr; /* structure to hold elapsed time */

  /* parse the command line */
  if (argc != 3) {
    usage(argv[0]);
  } else {
    param_file = argv[1];
    obstacle_file = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values
   * from file */
  gettimeofday(&timstr, NULL);
  double tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  double init_tic = tot_tic;
  initialise(param_file, obstacle_file, &params, &src_cells, &dst_cells,
             &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  double init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  double comp_tic = init_toc;

  for (int tt = 0; tt < params.max_iters; tt += 2) {
    av_vels[tt] = timestep(params, src_cells, dst_cells, obstacles);
    av_vels[tt + 1] = timestep(params, dst_cells, src_cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, src_cells));
#endif
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  double comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  double col_tic = comp_toc;

  // Collate data from ranks here

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  double col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  double tot_toc = col_toc;

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n",
         calc_reynolds(params, src_cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
  write_values(params, src_cells, obstacles, av_vels);
  finalise(&params, &src_cells, &dst_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed *src_cells, t_speed *dst_cells,
               bool *obstacles) {
  int tot_cells = 0; /* no. of src_cells used in calculation */
  float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

  /* ACCELERATE FLOW */
  /* compute weighting factors */
  const float accel_w1 = params.density * params.accel / 9.f;
  const float accel_w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;

#pragma omp parallel for schedule(static)
  for (int ii = 0; ii < params.nx; ++ii) {
    const int offset = ii + jj * params.nx;
    /* if the cell is not occupied and
    ** we don't send a negative density */
    const bool condition = !obstacles[offset] &&
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

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

/* COMBINED PROPROGATE AND REBOUND */
/* loop over the src_cells in the grid */
#pragma omp parallel for reduction(+ : tot_cells, tot_u) schedule(static)
  for (int jj = 0; jj < params.ny; ++jj) {
    const int y_n = (jj + 1) % params.ny;
    const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
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
  dst_speed = (obstacles[offset])                                              \
                  ? src_speed_if_obstacle                                      \
                  : (src_speed_if_not_obstacle +                               \
                     params.omega * (d_equ[i] - src_speed_if_not_obstacle))

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
      /* increase counter of inspected src_cells */
      tot_cells += (obstacles[offset]) ? 0 : 1;
    }
  }
  return tot_u / (float)tot_cells;
}

float av_velocity(const t_param params, t_speed *src_cells, bool *obstacles) {
  int tot_cells = 0; /* no. of src_cells used in calculation */
  float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

#pragma omp parallel for reduction(+ : tot_cells, tot_u) schedule(static)
  /* loop over all non-blocked src_cells */
  for (int jj = 0; jj < params.ny; ++jj) {
#pragma omp simd
    for (int ii = 0; ii < params.nx; ++ii) {
      /* ignore occupied src_cells */
      const int offset = ii + jj * params.nx;

      /* local density total */
      const float local_density =
          src_cells->speed0[offset] + src_cells->speed1[offset] +
          src_cells->speed2[offset] + src_cells->speed3[offset] +
          src_cells->speed4[offset] + src_cells->speed5[offset] +
          src_cells->speed6[offset] + src_cells->speed7[offset] +
          src_cells->speed8[offset];

      /* x-component of velocity */
      const float u_x =
          (src_cells->speed1[offset] + src_cells->speed5[offset] +
           src_cells->speed8[offset] -
           (src_cells->speed3[offset] + src_cells->speed6[offset] +
            src_cells->speed7[offset])) /
          local_density;
      /* compute y velocity component */
      const float u_y =
          (src_cells->speed2[offset] + src_cells->speed5[offset] +
           src_cells->speed6[offset] -
           (src_cells->speed4[offset] + src_cells->speed7[offset] +
            src_cells->speed8[offset])) /
          local_density;
      /* accumulate the norm of x- and y- velocity components */
      tot_u += (obstacles[offset]) ? 0 : sqrtf((u_x * u_x) + (u_y * u_y));
      /* increase counter of inspected src_cells */
      tot_cells += (obstacles[offset]) ? 0 : 1;
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char *paramfile, const char *obstaclefile,
               const t_param *params, t_speed **src_cells_ptr,
               t_speed **dst_cells_ptr, bool **obstacles_ptr,
               float **av_vels_ptr) {
  char message[1024]; /* message buffer */

  int xx, yy;  /* generic array indices */
  int blocked; /* indicates whether a cell is blocked by an obstacle */

  /* open the parameter file */
  FILE *fp = fopen(paramfile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  int retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1)
    die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1)
    die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->max_iters));

  if (retval != 1)
    die("could not read param file: maxIters", __LINE__, __FILE__);

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

  /* main grid */
  /* we use malloc here because only the arrays within the structure need to be
   * aligned */
  *src_cells_ptr = (t_speed *)malloc(sizeof(t_speed));
  initialise_speed(params, *src_cells_ptr);

  if (*src_cells_ptr == NULL)
    die("cannot allocate memory for src_cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *dst_cells_ptr = (t_speed *)malloc(sizeof(t_speed));
  initialise_speed(params, *dst_cells_ptr);

  if (*dst_cells_ptr == NULL)
    die("cannot allocate memory for dst_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr =
      (bool *)aligned_alloc(32, sizeof(int8_t) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density / 9.f;
  const float w2 = params->density / 36.f;

#pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params->ny; ++jj) {
#pragma omp simd
    for (int ii = 0; ii < params->nx; ++ii) {
      const int offset = ii + jj * params->nx;
      /* centre */
      (*src_cells_ptr)->speed0[offset] = w0;
      /* axis directions */
      (*src_cells_ptr)->speed1[offset] = w1;
      (*src_cells_ptr)->speed2[offset] = w1;
      (*src_cells_ptr)->speed3[offset] = w1;
      (*src_cells_ptr)->speed4[offset] = w1;
      /* diagonals */
      (*src_cells_ptr)->speed5[offset] = w2;
      (*src_cells_ptr)->speed6[offset] = w2;
      (*src_cells_ptr)->speed7[offset] = w2;
      (*src_cells_ptr)->speed8[offset] = w2;
    }
  }

/* first set all src_cells in obstacle array to zero */
#pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params->ny; ++jj) {
#pragma omp simd
    for (int ii = 0; ii < params->nx; ++ii) {
      (*obstacles_ptr)[ii + jj * params->nx] = false;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked src_cells list */
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
    (*obstacles_ptr)[xx + yy * params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float *)aligned_alloc(32, sizeof(float) * params->max_iters);

  return EXIT_SUCCESS;
}

inline void initialise_speed(const t_param *params, t_speed *speed) {
  /* we align each array */
  speed->speed0 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
  speed->speed1 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
  speed->speed2 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
  speed->speed3 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
  speed->speed4 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
  speed->speed5 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
  speed->speed6 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
  speed->speed7 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
  speed->speed8 =
      (float *)aligned_alloc(32, sizeof(float) * (params->ny * params->nx));
}

int finalise(const t_param *params, t_speed **src_cells_ptr,
             t_speed **dst_cells_ptr, bool **obstacles_ptr,
             float **av_vels_ptr) {
  /*
  ** free up allocated memory
  */
  finalise_speed(*src_cells_ptr);
  free(*src_cells_ptr);
  *src_cells_ptr = NULL;

  finalise_speed(*dst_cells_ptr);
  free(*dst_cells_ptr);
  *dst_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

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

float calc_reynolds(const t_param params, t_speed *src_cells, bool *obstacles) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, src_cells, obstacles) * params.reynolds_dim /
         viscosity;
}

float total_density(const t_param params, t_speed *src_cells) {
  float total = 0.f; /* accumulator */

#pragma omp parallel for reduction(+ : total) schedule(static)
  for (int jj = 0; jj < params.ny; ++jj) {
#pragma omp simd
    for (int ii = 0; ii < params.nx; ++ii) {
      const int offset = ii + jj * params.nx;
      total += src_cells->speed0[offset];
      total += src_cells->speed1[offset] + src_cells->speed2[offset] +
               src_cells->speed3[offset] + src_cells->speed4[offset];
      total += src_cells->speed5[offset] + src_cells->speed6[offset] +
               src_cells->speed7[offset] + src_cells->speed8[offset];
    }
  }

  return total;
}

int write_values(const t_param params, const t_speed *src_cells,
                 const bool *obstacles, const float *av_vels) {
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
        local_density = src_cells->speed0[offset] + src_cells->speed1[offset] +
                        src_cells->speed2[offset] + src_cells->speed3[offset] +
                        src_cells->speed4[offset] + src_cells->speed5[offset] +
                        src_cells->speed6[offset] + src_cells->speed7[offset] +
                        src_cells->speed8[offset];

        /* x-component of velocity */
        const float u_x =
            (src_cells->speed1[offset] + src_cells->speed5[offset] +
             src_cells->speed8[offset] -
             (src_cells->speed3[offset] + src_cells->speed6[offset] +
              src_cells->speed7[offset])) /
            local_density;
        /* compute y velocity component */
        const float u_y =
            (src_cells->speed2[offset] + src_cells->speed5[offset] +
             src_cells->speed6[offset] -
             (src_cells->speed4[offset] + src_cells->speed7[offset] +
              src_cells->speed8[offset])) /
            local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u,
              pressure, obstacles[ii + params.nx * jj]);
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
