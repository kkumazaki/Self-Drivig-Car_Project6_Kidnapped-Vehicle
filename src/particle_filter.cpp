/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  //(1)Set the number of particles.
  num_particles = 100; 
  //num_particles = 50; 

  //(2)Initialize all particles to first position and all weights to 1. 
  //   Add random Gaussian noise to each particle.
  std::default_random_engine gen; 

  //normal_distribution<double> dist_x(x, std[0]);
  //normal_distribution<double> dist_y(y, std[1]);
  //normal_distribution<double> dist_theta(theta, std[2]);
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);

  for (int i = 0; i < num_particles; i++){
    Particle p;
    p.id = i; //ID of particles
    p.x = x + dist_x(gen); //random x position of particles
    p.y = y + dist_y(gen); //random y position of particles
    p.theta = theta + dist_theta(gen); //random theta of particles
    p.weight = 1.0; //wight of particles
    particles.push_back(p); 
  }

  //double test_x = particles[0].x;
  //double test_y = particles[0].y;
  //double test_theta = particles[0].theta;
  //double test_wight = particles[0].weight;

  //(3)Initialize only once.
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  //(1)Add measurements to each particle and add random Gaussian noise.
  // When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  std::default_random_engine gen; 

  for (int i = 0; i < num_particles; i++) {
    if(fabs(yaw_rate)> 0.001){
      particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t)- sin(particles[i].theta));
      particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    else{
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
     
    //normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    //normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    //normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    particles[i].x += dist_x(gen); //random x position of particles
    particles[i].y += dist_y(gen); //random y position of particles
    particles[i].theta += dist_theta(gen); //random theta of particles
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  //(1)Nearest Neighber:
  //   Find the predicted measurement that is closest to each observed measurement and assign the observed measurement to this particular landmark.
  for (unsigned int i = 0; i < observations.size(); i ++){
      double min_dist = std::numeric_limits<double>::max();
      int map_id = -1;
      for (unsigned int j = 0; j < predicted.size(); j++){
        double dist_current = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

        if (min_dist > dist_current){
          min_dist = dist_current;
          map_id = predicted[j].id;
        }
      }
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */


  for (int i = 0; i <num_particles; i++){
    //(1)Landmarks that are predicted by sensor range
    vector<LandmarkObs> predicted;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      if(dist(particles[i].x, particles[i].y, landmark_x, landmark_y) < sensor_range){
        predicted.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    //(2)Transfer from vehicle coordinate to map coordinate   
    vector<LandmarkObs> transformed_Obs;
    for(unsigned int k = 0; k < observations.size(); k++){
      double xm = cos(particles[k].theta)*observations[k].x - sin(particles[k].theta)*observations[k].y + particles[k].x;
      double ym = sin(particles[k].theta)*observations[k].x + cos(particles[k].theta)*observations[k].y + particles[k].y;
      transformed_Obs.push_back(LandmarkObs{observations[k].id, xm, ym});
    }
    //(3)Match subset of landmarks
    dataAssociation(predicted, transformed_Obs);

    //(4)Calculate weights
		particles[i].weight = 1.0;
    
    for (unsigned int l = 0; l < transformed_Obs.size(); l++){
      int index = 0;
      while(transformed_Obs[l].id != predicted[index].id){
        index++;
      }
      particles[i].weight *= 1./(2.*M_PI*std_landmark[0]*std_landmark[1])
                            *exp(-1.*(pow(transformed_Obs[l].x-predicted[index].x,2)/(2*std_landmark[0]*std_landmark[0])
                            + (pow(transformed_Obs[l].y-predicted[index].y,2)/(2*std_landmark[1]*std_landmark[1]))));
    }

  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	//(1)Get an array of weights for a discrete distribution
	vector<double> weights;  	
	std::default_random_engine gen;
	vector<Particle> new_particles;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	//(2)Resampling
	for (int i = 0; i < num_particles; i++) {
		std::discrete_distribution<> dst2(weights.begin(), weights.end());
		new_particles.push_back(particles[dst2(gen)]);
	}
	particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}