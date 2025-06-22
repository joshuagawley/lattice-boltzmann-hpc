#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

/* target OpenCl 3.0 */
#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define OCLFILE "kernels.cl"

#define LOCAL_SIZE_X 32
#define LOCAL_SIZE_Y 4

/* struct to hold the parameter values */
typedef struct {
  int nx;              /* no. of cells in x-direction */
  int ny;              /* no. of cells in y-direction */
  int max_iters;       /* no. of iterations */
  int reynolds_dim;    /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  int unblocked_cells; /* total unblocked cells */
} t_param;

/* struct to hold OpenCL objects */
typedef struct {
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;

  cl_program program;
  cl_kernel accelerate_flow;
  cl_kernel compute_src_to_dst;
  cl_kernel compute_dst_to_src;
  cl_kernel reduce;

  cl_mem src_cells;
  cl_mem dst_cells;
  cl_mem obstacles;

  cl_mem vels;
  cl_mem av_vels;

  int work_groups;
} t_ocl;

/* function prototypes */

/* load params, allocate memory, load obstacles & initialise fluid particle
 * densities
 */
int initialise(const char *paramfile, const char *obstaclefile, t_param *params,
               float **src_cells_ptr, float **dst_cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr, t_ocl *ocl);

int timestep(const t_param params, t_ocl ocl, bool src_to_dst);
int accelerate_flow(const t_param params, t_ocl ocl, bool src_to_dst);
float compute(const t_param params, t_ocl ocl, bool src_to_dst);
int write_values(const t_param params, float *cells, int *obstacles,
                 float *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, float **src_cells_ptr,
             float **dst_cells_ptr, int **obstacles_ptr, float **av_vels_ptr,
             t_ocl ocl);

/* Sum all the densities in the grid. */
/* The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float *cells);

/* compute average velocity */
float av_velocity(const t_param params, const float *cells,
                  const int *obstacles, t_ocl ocl);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float *cells, int *obstacles,
                    t_ocl ocl);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

cl_device_id selectOpenCLDevice();

int main(int argc, char *argv[]) {
  char *paramfile = NULL;    /* name of the input parameter file        */
  char *obstaclefile = NULL; /* name of a the input obstacle file       */
  t_param params;            /* struct to hold parameter values         */
  t_ocl ocl;                 /* struct to hold OpenCL objects           */
  float *src_cells = NULL;   /* grid containing fluid densities         */
  float *dst_cells = NULL;   /* scratch space                           */
  int *obstacles = NULL;     /* grid indicating which cells are blocked */
  float *av_vels =
      NULL; /* a record of the av. velocity computed for each timestep */
  cl_int err;
  struct timeval timstr; /* structure to hold elapsed time              */
  struct rusage ru;      /* structure to hold CPU time-- system and user */
  double tic,
      toc; /* floating point numbers to calculate elapsed wallclock time */
  double usrtim; /* floating point number to record elapsed user CPU time */
  double systim; /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3) {
    usage(argv[0]);
  } else {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &src_cells, &dst_cells,
             &obstacles, &av_vels, &ocl);

  const int grid_size = params.nx * params.ny;

  /* iterate for max_iters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* Write cells to OpenCL buffer */
  err = clEnqueueWriteBuffer(ocl.queue, ocl.src_cells, CL_TRUE, 0,
                             sizeof(float) * grid_size * NSPEEDS, src_cells, 0,
                             NULL, NULL);
  checkError(err, "writing src_cells data", __LINE__);

  /* Write obstacles to OpenCL buffer*/
  err = clEnqueueWriteBuffer(ocl.queue, ocl.obstacles, CL_TRUE, 0,
                             sizeof(cl_int) * grid_size, obstacles, 0, NULL,
                             NULL);
  checkError(err, "writing obstacles data", __LINE__);

  for (int tt = 0; tt < params.max_iters; ++tt) {
    /* Run with src_cells -> dst_cells */
    err = clSetKernelArg(ocl.compute_src_to_dst, 10, sizeof(cl_int), &tt);
    checkError(err, "setting compute_src_to_dst arg 10", __LINE__);
    timestep(params, ocl, true);

    ++tt;

    /* Run with dst_cells -> src_cells */
    err = clSetKernelArg(ocl.compute_dst_to_src, 10, sizeof(cl_int), &tt);
    checkError(err, "setting compute_dst_to_src arg 10", __LINE__);
    timestep(params, ocl, false);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, src_cells));
#endif
  }

  /* Enqueue reduce kernel */
  size_t reduce_global[1] = {params.max_iters};
  size_t reduce_local[1] = {1};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.reduce, 1, NULL, reduce_global,
                               reduce_local, 0, NULL, NULL);
  checkError(err, "enqueueing reduce kernel", __LINE__);

  /* Read src_cells from device */
  err = clEnqueueReadBuffer(ocl.queue, ocl.src_cells, CL_TRUE, 0,
                            sizeof(float) * grid_size * NSPEEDS, src_cells, 0,
                            NULL, NULL);
  checkError(err, "reading src_cells data", __LINE__);

  /* Read av_vels from device */
  err = clEnqueueReadBuffer(ocl.queue, ocl.av_vels, CL_TRUE, 0,
                            sizeof(float) * params.max_iters, av_vels, 0, NULL,
                            NULL);
  checkError(err, "reading dst_cells data", __LINE__);

  /* Wait for kernel to finish */
  err = clFinish(ocl.queue);
  checkError(err, "waiting for final kernel", __LINE__);

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n",
         calc_reynolds(params, src_cells, obstacles, ocl));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, src_cells, obstacles, av_vels);
  finalise(&params, &src_cells, &dst_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_ocl ocl, bool src_to_dst) {
  cl_int err;

  /* Enqueue computation kernel */
  size_t computation_global[2] = {params.nx, params.ny};
  size_t local[2] = {LOCAL_SIZE_X, LOCAL_SIZE_Y};
  err = clEnqueueNDRangeKernel(
      ocl.queue, (src_to_dst) ? ocl.compute_src_to_dst : ocl.compute_dst_to_src,
      2, NULL, computation_global, local, 0, NULL, NULL);
  checkError(err, "enqueueing computation kernel", __LINE__);

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, const float *cells,
                  const int *obstacles, t_ocl ocl) {
  int tot_cells = 0; /* no. of src_cells used in calculation */
  float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

  const int grid_size = params.nx * params.ny;

  /* loop over all non-blocked cells*/
  for (int jj = 0; jj < params.ny; ++jj) {
    for (int ii = 0; ii < params.nx; ++ii) {
      const int offset = ii + jj * params.nx;

      /* local density total */
      const float local_density =
          cells[(0 * grid_size) + offset] + cells[(1 * grid_size) + offset] +
          cells[(2 * grid_size) + offset] + cells[(3 * grid_size) + offset] +
          cells[(4 * grid_size) + offset] + cells[(5 * grid_size) + offset] +
          cells[(6 * grid_size) + offset] + cells[(7 * grid_size) + offset] +
          cells[(8 * grid_size) + offset];

      /* x-component of velocity */
      const float u_x =
          (cells[(1 * grid_size) + offset] + cells[(5 * grid_size) + offset] +
           cells[(8 * grid_size) + offset] -
           (cells[(3 * grid_size) + offset] + cells[(6 * grid_size) + offset] +
            cells[(7 * grid_size) + offset])) /
          local_density;
      /* compute y velocity component */
      const float u_y =
          (cells[(2 * grid_size) + offset] + cells[(5 * grid_size) + offset] +
           cells[(6 * grid_size) + offset] -
           (cells[(4 * grid_size) + offset] + cells[(7 * grid_size) + offset] +
            cells[(8 * grid_size) + offset])) /
          local_density;
      /* accumulate the norm of x- and y- velocity components */
      tot_u += (obstacles[offset]) ? 0 : sqrtf((u_x * u_x) + (u_y * u_y));
      /* increase counter of inspected src_cells */
      tot_cells += (obstacles[offset]) ? 0 : 1;
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char *paramfile, const char *obstaclefile, t_param *params,
               float **src_cells_ptr, float **dst_cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr, t_ocl *ocl) {
  char message[1024]; /* message buffer */
  FILE *fp;           /* file pointer   */
  int xx, yy;         /* generic array indices */
  int blocked;        /* indicates whether a cell is blocked by an obstacle */
  int retval;         /* to hold return value for checking */
  char *ocl_src;      /* OpenCL kernel source */
  long ocl_size;      /* size of OpenCL kernel source */

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

  const int grid_size = params->nx * params->ny;

  /* main grid */
  *src_cells_ptr = malloc(sizeof(float) * NSPEEDS * grid_size);
  if (*src_cells_ptr == NULL)
    die("cannot allocate memory for src_cells speeds", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *dst_cells_ptr = malloc(sizeof(float) * NSPEEDS * grid_size);
  if (*dst_cells_ptr == NULL)
    die("cannot allocate memory for dst_cells speeds", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * grid_size);
  if (*obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density / 9.f;
  const float w2 = params->density / 36.f;

  for (int jj = 0; jj < params->ny; ++jj) {
    for (int ii = 0; ii < params->nx; ++ii) {
      const int offset = ii + jj * params->nx;
      /* centre */
      (*src_cells_ptr)[(0 * grid_size) + offset] = w0;
      /* axis directions */
      for (int kk = 1; kk < 5; ++kk) {
        (*src_cells_ptr)[(kk * grid_size) + offset] = w1;
      }
      /* diagonals */
      for (int kk = 5; kk < 9; ++kk) {
        (*src_cells_ptr)[(kk * grid_size) + offset] = w2;
      }
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; ++jj) {
    for (int ii = 0; ii < params->nx; ++ii) {
      (*obstacles_ptr)[ii + jj * params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL) {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  params->unblocked_cells = grid_size;

  /* read-in the blocked cells list */
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

    params->unblocked_cells--;
  }

  /* and close the file */
  fclose(fp);

  /* allocate space to hold a record of the avarage vels computed at each
   * timestep
   */
  *av_vels_ptr = (float *)malloc(sizeof(float) * params->max_iters);

  cl_int err;

  ocl->device = selectOpenCLDevice();

  /* Create OpenCL context */
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL) {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  /* Create OpenCL command queue */
  ocl->queue =
      clCreateCommandQueueWithProperties(ocl->context, ocl->device, NULL, &err);
  checkError(err, "creating command queue", __LINE__);

  /* Load OpenCL kernel source */
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char *)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  /* Create OpenCL program */
  ocl->program = clCreateProgramWithSource(ocl->context, 1,
                                           (const char **)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  /* Build OpenCL program */
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t sz;
    clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0,
                          NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, sz,
                          buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  /* Create OpenCL kernels */
  ocl->compute_src_to_dst = clCreateKernel(ocl->program, "computation", &err);
  checkError(err, "creating compute_src_to_dst kernel", __LINE__);

  ocl->compute_dst_to_src = clCreateKernel(ocl->program, "computation", &err);
  checkError(err, "creating compute_dst_to_src kernel", __LINE__);

  ocl->reduce = clCreateKernel(ocl->program, "reduce", &err);
  checkError(err, "creating reduce kernel", __LINE__);

  /* Allocate OpenCL buffers */
  ocl->src_cells =
      clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                     sizeof(float) * grid_size * NSPEEDS, NULL, &err);
  checkError(err, "creating src_cells buffer", __LINE__);
  ocl->dst_cells =
      clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                     sizeof(float) * grid_size * NSPEEDS, NULL, &err);
  checkError(err, "creating dst_cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
                                  sizeof(cl_int) * grid_size, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);

  /* Divide grid into workgroups based on local size setting*/
  ocl->work_groups = (grid_size) / (LOCAL_SIZE_X * LOCAL_SIZE_Y);

  ocl->vels = clCreateBuffer(
      ocl->context, CL_MEM_READ_WRITE,
      sizeof(float) * ocl->work_groups * params->max_iters, NULL, &err);
  checkError(err, "creating vels buffer", __LINE__);

  ocl->av_vels = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
                                sizeof(float) * params->max_iters, NULL, &err);
  checkError(err, "creating av_vels buffer", __LINE__);

  float tot_cells = (float)params->unblocked_cells;

  err = clSetKernelArg(ocl->reduce, 0, sizeof(cl_mem), &ocl->vels);
  checkError(err, "setting reduce arg 0", __LINE__);
  err = clSetKernelArg(ocl->reduce, 1, sizeof(cl_mem), &ocl->av_vels);
  checkError(err, "setting reduce arg 1", __LINE__);
  err = clSetKernelArg(ocl->reduce, 2, sizeof(cl_int), &tot_cells);
  checkError(err, "setting reduce arg 2", __LINE__);
  err = clSetKernelArg(ocl->reduce, 3, sizeof(cl_int), &ocl->work_groups);
  checkError(err, "setting reduce arg 3", __LINE__);

  err = clSetKernelArg(ocl->compute_src_to_dst, 0, sizeof(cl_mem),
                       &ocl->src_cells);
  checkError(err, "setting compute_src_to_dst arg 0", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 1, sizeof(cl_mem),
                       &ocl->dst_cells);
  checkError(err, "setting compute_src_to_dst arg 1", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 2, sizeof(cl_mem),
                       &ocl->obstacles);
  checkError(err, "setting compute_src_to_dst arg 2", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 3, sizeof(cl_int), &params->nx);
  checkError(err, "setting compute_src_to_dst arg 3", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 4, sizeof(cl_int), &params->ny);
  checkError(err, "setting compute_src_to_dst arg 4", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 5, sizeof(cl_float),
                       &params->omega);
  checkError(err, "setting compute_src_to_dst arg 5", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 6, sizeof(cl_mem), &ocl->vels);
  checkError(err, "setting compute_src_to_dst arg 6", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 7,
                       sizeof(float) * LOCAL_SIZE_X * LOCAL_SIZE_Y, NULL);
  checkError(err, "setting compute_src_to_dst arg 7", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 8, sizeof(cl_float),
                       &params->density);
  checkError(err, "setting compute_src_to_dst arg 8", __LINE__);
  err = clSetKernelArg(ocl->compute_src_to_dst, 9, sizeof(cl_float),
                       &params->accel);
  checkError(err, "setting compute_src_to_dst arg 9", __LINE__);

  err = clSetKernelArg(ocl->compute_dst_to_src, 0, sizeof(cl_mem),
                       &ocl->dst_cells);
  checkError(err, "setting compute_dst_to_src arg 0", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 1, sizeof(cl_mem),
                       &ocl->src_cells);
  checkError(err, "setting compute_dst_to_src arg 1", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 2, sizeof(cl_mem),
                       &ocl->obstacles);
  checkError(err, "setting compute_dst_to_src arg 2", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 3, sizeof(cl_int), &params->nx);
  checkError(err, "setting compute_dst_to_src arg 3", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 4, sizeof(cl_int), &params->ny);
  checkError(err, "setting compute_dst_to_src arg 4", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 5, sizeof(cl_float),
                       &params->omega);
  checkError(err, "setting compute_dst_to_src arg 5", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 6, sizeof(cl_mem), &ocl->vels);
  checkError(err, "setting compute_dst_to_src arg 6", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 7,
                       sizeof(float) * LOCAL_SIZE_X * LOCAL_SIZE_Y, NULL);
  checkError(err, "setting compute_dst_to_src arg 7", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 8, sizeof(cl_float),
                       &params->density);
  checkError(err, "setting compute_dst_to_src arg 8", __LINE__);
  err = clSetKernelArg(ocl->compute_dst_to_src, 9, sizeof(cl_float),
                       &params->accel);
  checkError(err, "setting compute_dst_to_src arg 9", __LINE__);

  return EXIT_SUCCESS;
}

int finalise(const t_param *params, float **src_cells_ptr,
             float **dst_cells_ptr, int **obstacles_ptr, float **av_vels_ptr,
             t_ocl ocl) {
  /*
   ** free up allocated memory
   */
  free(*src_cells_ptr);
  *src_cells_ptr = NULL;

  free(*dst_cells_ptr);
  *dst_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.src_cells);
  clReleaseMemObject(ocl.dst_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.compute_src_to_dst);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, float *cells, int *obstacles,
                    t_ocl ocl) {
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles, ocl) * params.reynolds_dim /
         viscosity;
}

float total_density(const t_param params, float *cells) {
  /* accumulator */
  float total = 0.f;

  const int grid_size = params.nx * params.ny;

  for (int jj = 0; jj < params.ny; ++jj) {
    for (int ii = 0; ii < params.nx; ++ii) {
      for (int kk = 0; kk < NSPEEDS; ++kk) {
        const int offset = ii + jj * params.nx;
        total += cells[(kk * grid_size) + offset];
      }
    }
  }

  return total;
}

int write_values(const t_param params, float *cells, int *obstacles,
                 float *av_vels) {
  FILE *fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  const int grid_size = params.nx * params.ny;
  float local_density; /* per grid cell sum of densities */
  float pressure;      /* fluid pressure in grid cell */
  float u_x;           /* x-component of velocity in grid cell */
  float u_y;           /* y-component of velocity in grid cell */
  float u;             /* norm--root of summed squares--of u_x and u_y */

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

      } else { /* no obstacle */
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; ++kk) {
          local_density += cells[(kk * grid_size) + (offset)];
        }

        /* compute x velocity component */
        u_x =
            (cells[(1 * grid_size) + offset] + cells[(5 * grid_size) + offset] +
             cells[(8 * grid_size) + offset] -
             (cells[(3 * grid_size) + offset] +
              cells[(6 * grid_size) + offset] +
              cells[(7 * grid_size) + offset])) /
            local_density;
        /* compute y velocity component */
        u_y =
            (cells[(2 * grid_size) + offset] + cells[(5 * grid_size) + offset] +
             cells[(6 * grid_size) + offset] -
             (cells[(4 * grid_size) + offset] +
              cells[(7 * grid_size) + offset] +
              cells[(8 * grid_size) + offset])) /
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

void checkError(cl_int err, const char *op, const int line) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
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

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice() {
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  /* Get list of platforms */
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  /* Get list of devices */
  for (cl_uint p = 0; p < num_platforms; ++p) {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES - total_devices, devices + total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  /* Print list of devices */
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; ++d) {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  /* Use first device unless OCL_DEVICE environment variable used */
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env) {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices) {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  /* Print OpenCL device name */
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME, MAX_DEVICE_NAME, name,
                  NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}