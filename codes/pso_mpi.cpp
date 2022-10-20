#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>
#include <time.h>
#include <sys/time.h>
// #include <omp.h> 
#include <mpi.h>
//#include "hdf5.h"

using namespace std;

#define MASTER_RANK 0

double inf = std::numeric_limits<double>::infinity();
const int N=20,dim=2,Npoch=100;
double lower_bounds[2] = {-50,-50};
double upper_bounds[2] = {50,50};
double **points, **score_localbest, **pos_localbest, score_gbest, **pos_gbest, **velocities, **scores;


double get_wtime(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double clip(double value, double low, double high){
  if (value > high){
    value = high;
  } else if (value < low){
    value = low;
  }
  return value;
}

double get_random(double min, double max){
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(min, max);
  
  return dist(mt);
}

void print_array(double** arr, int dimension, int population){
  for (int i=0; i<dimension; i++){
    for (int j=0; j<population; j++){
      cout << arr[i][j] << " ";
    }
    cout << endl;
  }
}

double param_to_real(double value, double lower_bound, double upper_bound){
  value = lower_bound + (upper_bound - lower_bound)*value;
  return value;
}

double** random_array(double min, double max, int dimension, int population){
  double** points = new double*[dimension];
  for (int i =0; i< dimension; i++ ){
    points[i] = new double[population];
    for (int j=0; j<population; j++){
      points[i][j] = get_random(min,max);
    }
  }
  return points;
}

double** infs(int dimension, int population){
  double** points = new double*[dimension];
  for (int i =0; i< dimension; i++ ){
    points[i] = new double[population];
    for (int j=0; j<population; j++){
      points[i][j] = inf;
    }
  }
  return points;
}

double** zeros(int dimension, int population){
  double** points = new double*[dimension];
  for (int i =0; i< dimension; i++ ){
    points[i] = new double[population];
    for (int j=0; j<population; j++){
      points[i][j] = 0.0;
    }
  }
  return points;
}

double** ones(int dimension, int population){
  double** points = new double*[dimension];
  for (int i =0; i< dimension; i++ ){
    points[i] = new double[population];
    for (int j=0; j<population; j++){
      points[i][j] = 1.0;
    }
  }
  return points;
}

double fitness(double xvalue, double yvalue){
  double fitness_value = 0.0;
  // Himmelblau's function
  fitness_value = pow((pow(xvalue,2) + yvalue -11),2) + pow((xvalue + pow(yvalue,2) - 7 ),2);
  return fitness_value;
}

void output(){
//   printf("points (%d, %d):\n",dim,N);
//   print_array(points,dim,N);
//   
//   printf("pos localbest (%d, %d):\n",dim,N);
//   print_array(pos_localbest,dim,N);
//   
//   printf("velocities (%d, %d):\n",dim,N);
//   print_array(velocities,dim,N);
//   
//   printf("score localbest (%d, %d):\n",1,N);
//   print_array(score_localbest,1,N);
//   
//   printf("scores (%d, %d):\n",1,N);
//   print_array(scores,1,N);
//   
//   printf("score gbest: %f\n",score_gbest);
//   
  printf("pos gbest (%d, %d):\n",dim,1);
  print_array(pos_gbest,dim,1);
  
  printf("***** best optimization score *****\n");
  printf("%f\n",score_gbest);
  
  printf("***** optimized results (parameters) *****\n");
  double x,y;
  x = param_to_real(pos_gbest[0][0],lower_bounds[0],upper_bounds[0]);
  y = param_to_real(pos_gbest[1][0],lower_bounds[1],upper_bounds[1]);
  printf("x %f, y %f\n",x,y);
}

void init_pso(){

  points = random_array(0.0,1.0,dim,N);
  score_localbest = infs(1,N);
  scores = infs(1,N);
  pos_localbest = zeros(dim,N);
  score_gbest = inf;
  pos_gbest = zeros(dim,1);
  velocities = random_array(-1.0,1.0,dim,N);
  
  printf("\nInitialize points...\n");
  output();
  
  double x,y;
  for (int j=0; j<N; j++){
    x = param_to_real(points[0][j],lower_bounds[0],upper_bounds[0]);
    y = param_to_real(points[1][j],lower_bounds[1],upper_bounds[1]);
    scores[0][j] = fitness(x,y);
  }  

}


void update_particles(int processor_rank, int number_particles){
  int start = processor_rank*number_particles;
  int end = start + number_particles;
  
  if (start >= N) {
    return;
  } else if (end >= N) {
    end = N;
  }
  
  double **new_points, **new_velocities;
  new_points = random_array(0.0,1.0,dim,N);
  new_velocities = random_array(-1.0,1.0,dim,N);
  
  //coefficients 
  double c1 = 1.494; //suggested by paper Eberhart and Shi (2001)
  double v_max = 1.0;
  double Rcoeff1, Rcoeff2, viscous_factor;  
  
  //random coefficient between 0 and 1
  Rcoeff1 = get_random(0.0,1.0);
  Rcoeff2 = get_random(0.0,1.0);
  viscous_factor = 0.5 + 0.5*get_random(0.0,1.0);

// update n-th particle for each MPI rank 
  for (int j=start; j<end; j++){
    for (int i=0; i<dim; i++){
      new_velocities[i][j] = velocities[i][j]*viscous_factor \
                       + c1* Rcoeff1 * (pos_localbest[i][j] - points[i][j]) \
                       + c1* Rcoeff2 * (pos_gbest[i][j] - points[i][j]);
      new_velocities[i][j] = clip(new_velocities[i][j],-v_max,v_max);
      new_points[i][j] = points[i][j] + new_velocities[i][j];
      if (new_points[i][j] > 1.0){
        new_points[i][j] = 1.0;
        new_velocities[i][j] = 0.0;
      } else if (new_points[i][j] < 0.0){
        new_points[i][j] = 0.0;
        new_velocities[i][j] = 0.0;
      }
      points[i][j] = new_points[i][j];
      velocities[i][j] = new_velocities[i][j];
    }
    double x,y; 
    x = param_to_real(points[0][j],lower_bounds[0],upper_bounds[0]);
    y = param_to_real(points[1][j],lower_bounds[1],upper_bounds[1]);
    scores[0][j] = fitness(x,y);
  }

}


void update_best(){
// #pragma omp parallel for schedule(static)  // It is not safe to do comparison and update\
with shared memory threading. It is also not easy to do distributed memory parallelization. \
Because it is difficult to keep track of particle index.
  for (int j=0; j<N; j++){
    if (scores[0][j] < score_localbest[0][j]){
      score_localbest[0][j] = scores[0][j];
      for (int i=0; i<dim; i++){
        pos_localbest[i][j] = points[i][j];
      }
    }
    
    if (scores[0][j] < score_gbest){
      score_gbest = scores[0][j];
      for (int i=0; i<dim; i++){
        pos_gbest[i][0] = points[i][j];
      }
    } 
  }
}


int main(int argc, char *argv[])
{
  int mpi_rank, mpi_ntasks;
  double time;
  
  MPI_Datatype aggregateType;
  
  init_pso();
  update_best();
  printf("\nAfter first run...\n");
  output(); 
  
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_ntasks);
  
  int particles_per_processor = (N + mpi_ntasks - 1) / mpi_ntasks;
  
  MPI_Type_contiguous(2, MPI_DOUBLE, &aggregateType);
  MPI_Type_commit(&aggregateType);
  
  MPI_Bcast(scores[0], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  MPI_Bcast(points[0], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  MPI_Bcast(points[1], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  MPI_Bcast(velocities[0], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  MPI_Bcast(velocities[1], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  MPI_Bcast(pos_localbest[0], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  MPI_Bcast(pos_localbest[1], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  MPI_Bcast(pos_gbest[0], 1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  MPI_Bcast(pos_gbest[1], 1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
  
  time = get_wtime();
  for (int i=0; i<Npoch; i++){
    
    update_particles(mpi_rank, particles_per_processor);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, scores[0], 1, aggregateType, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, points[0], 1, aggregateType, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, points[1], 1, aggregateType, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, velocities[0], 1, aggregateType, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, velocities[1], 1, aggregateType, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, pos_localbest[0], 1, aggregateType, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, pos_localbest[1], 1, aggregateType, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, pos_gbest[0], 1, aggregateType, MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, pos_gbest[1], 1, aggregateType, MPI_COMM_WORLD);
    
    if (mpi_rank == MASTER_RANK){
      update_best();
      MPI_Bcast(pos_localbest[0], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
      MPI_Bcast(pos_localbest[1], N, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
      MPI_Bcast(pos_gbest[0], 1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
      MPI_Bcast(pos_gbest[1], 1, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);
      printf("\nUpdate particles (iteration %d)...\n",i+1);
      output();
    }
  }
  
  time = get_wtime() - time;
  if (mpi_rank == MASTER_RANK){
    printf("time cost %f\n",time);
  }
  MPI_Finalize();
  
  return 0;
}

