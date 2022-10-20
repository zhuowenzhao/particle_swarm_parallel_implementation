#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
// #include <mpi.h>

using namespace std;

double inf = std::numeric_limits<double>::infinity();
const int N=20,dim=2,Npoch=100,loop_counts=10;;
double lower_bounds[2] = {-100,-100};
double upper_bounds[2] = {100,100};
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
#pragma omp parallel for schedule(static)
  for (int j=0; j<N; j++){
    x = param_to_real(points[0][j],lower_bounds[0],upper_bounds[0]);
    y = param_to_real(points[1][j],lower_bounds[1],upper_bounds[1]);
    scores[0][j] = fitness(x,y);
  }
  
}


void update_particles(){  
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
  
// update each particle 
#pragma omp parallel for schedule(static)
  for (int j=0; j<N; j++){
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
  }
  
  double x,y;
#pragma omp parallel for schedule(static)
  for (int j=0; j<N; j++){ 
    x = param_to_real(points[0][j],lower_bounds[0],upper_bounds[0]);
    y = param_to_real(points[1][j],lower_bounds[1],upper_bounds[1]);
    scores[0][j] = fitness(x,y);
  }
}


void update_best(){
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
  double time;  
  
//   time = get_wtime();
//   init_pso();
//   update_best();
//   printf("\nAfter first run...\n");
//   output();
//   int count=0;
//   while (score_gbest >= 0.01){
//     update_particles();
//     update_best();
//     printf("\nUpdate particles...\n");
//     output();
//     count += 1;
//   }
//   time = get_wtime() - time;
//   
//   printf("***** performance evaluation *****\n");
//   printf("time cost(s) %f total iterations %d\n",time,count);
  
  time = get_wtime();
  for (int i=0; i<loop_counts; i++){
    printf("\nParticles initiation (loop %d)...\n",i+1);
    init_pso();
    update_best();
    printf("\nAfter first run...\n");
    output();
    for (int j=0; j<Npoch; j++){
      update_particles();
      update_best();
      printf("\nUpdate particles (loop %d, iteration %d)...\n",i+1,j+1);
      output();
    }
  }
  time = get_wtime() - time;
  time /= loop_counts;
  printf("Average time cost %f\n",time);
  
  return 0;
}

