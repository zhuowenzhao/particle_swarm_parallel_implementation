#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>
//#include <string>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include "hdf5.h"
// #include <mpi.h>

using namespace std;

double inf = std::numeric_limits<double>::infinity();
const int N=20,dim=2,Npoch=2000;
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
    double x,y; 
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
  
  hid_t file_id;
  herr_t status; // error flag
  //open a file
  file_id = H5Fcreate("pso_results.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // arg are setting how the file is accessed.
  
  hsize_t dims[2]; //hdf5 size type
  dims[0] = dim;
  dims[1] = N;
  hid_t dataspace_id = H5Screate_simple(2, dims, NULL);  //rank is the array rank, size
  
  time = get_wtime();
  init_pso();
  update_best();
  printf("\nAfter first run...\n");
  output();
  for (int i=0; i<Npoch; i++){
    update_particles();
    update_best();
    printf("\nUpdate particles...\n");
    output();
    std::string name1 = "iter" + std::to_string(i) + "_particles";
    std::string name2 = "iter" + std::to_string(i) + "_velocities";
    const char * dname1 = name1.c_str();
    const char * dname2 = name2.c_str();
    
    hid_t dataset_id1   = H5Dcreate2(file_id, dname1, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_id2   = H5Dcreate2(file_id, dname2, H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    status = H5Dwrite(dataset_id1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, points);
    status = H5Dwrite(dataset_id2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, velocities);
    // herr_t H5Dwrite(hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t plist_id, const void *buf)
    status = H5Dclose(dataset_id1);
    status = H5Dclose(dataset_id2);
  }
  time = get_wtime() - time;
  status = H5Fclose(file_id);
  status = H5Sclose(dataspace_id); // why hdf5 not following asending order?
  
  printf("***** performance evaluation *****\n");
  printf("time cost(s) %f\n",time);
  
  return 0;
}

