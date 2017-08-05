/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;
double sigma_unc [3] = {0.3, 0.3, 0.01};

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 100;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for(int i=0;i<num_particles;i++){
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1; 
        particles.push_back(p);
    }
    
    is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
    
    for(int i=0;i<num_particles;i++){

        double x,y,theta;
        if (fabs(yaw_rate) < 0.00001) { //car going straight
            x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			theta = particles[i].theta;   
        }else{ //yaw rate is not equal to zero, car turning
            x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
            y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
        	theta = particles[i].theta + yaw_rate*delta_t;	
        }   

		//RANDOM GAUSSIAN NOISE
		// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
		//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
		//  http://www.cplusplus.com/reference/random/default_random_engine/
		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_theta(theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for(int i=0;i<observations.size();i++){
        double min_dist,dist;
        int index_min = -1;
        min_dist = INFINITY;

        for(int j=0;j<predicted.size();j++){
            
            double diff_x = predicted[j].x - observations[i].x;
            double diff_y = predicted[j].y - observations[i].y;
            double dist = sqrt(diff_x * diff_x + diff_y * diff_y); //euclidean distance

            if(dist<min_dist){
                min_dist = dist;
                index_min = predicted[j].id;
            }
        }     
        observations[i].id=index_min;

    }

}

//This function transform a local (car) observation into a global (map) coordinates
LandmarkObs transformation(LandmarkObs observation, Particle p){
    LandmarkObs local;
    
    local.id = observation.id;
    // Rotation performed in(3.33) : http://planning.cs.uiuc.edu/node99.html
    local.x = observation.x*cos(p.theta)-observation.y*sin(p.theta)+p.x;
    local.y = observation.x*sin(p.theta)+observation.y*cos(p.theta)+p.y;
        
    return local;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    //create a predicted_landmark vector
    for(int i=0;i<particles.size();i++){

        std::vector<LandmarkObs> predicted_landmarks;
        for(int j=0;j<map_landmarks.landmark_list.size();j++){

            //we have to consider only landmarks within the range of the sensor
            double diff_x, diff_y, dist;
            diff_x = map_landmarks.landmark_list[j].x_f - particles[i].x;
            diff_y = map_landmarks.landmark_list[j].y_f - particles[i].y;

 			if (fabs(diff_x) <= sensor_range && fabs(diff_y) <= sensor_range) 
                predicted_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f});
        }
    

        //before applying the association we have to transform the observations in the global coordinates
        std::vector<LandmarkObs> transformed_observations;
        for(int k=0;k<observations.size();k++)
            transformed_observations.push_back(transformation(observations[k],particles[i]));

        //associate the landmarks to the particles
        dataAssociation(predicted_landmarks,transformed_observations);
	
 		particles[i].weight = 1.0;

        //compute the probability

        for(int k=0;k<transformed_observations.size();k++){
            double obs_x,obs_y,l_x,l_y;
            obs_x = transformed_observations[k].x;
            obs_y = transformed_observations[k].y;
            
			//get the associated landmark
			for (unsigned int p = 0; p < predicted_landmarks.size(); p++) {
				if (transformed_observations[k].id == predicted_landmarks[p].id) {
					l_x = predicted_landmarks[p].x;
					l_y = predicted_landmarks[p].y;
				}
			}			

            double w = exp( -( pow(l_x-obs_x,2)/(2*pow(sigma_unc[0],2)) + pow(l_y-obs_y,2)/(2*pow(sigma_unc[1],2)) ) ) / ( 2*M_PI*sigma_unc[0]*sigma_unc[1] );
        	
			//update the weights
        	particles[i].weight = particles[i].weight*w;
        }

    }    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here. (wheel algorithm course)
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    uniform_int_distribution<int> dist_distribution(0,num_particles-1);
    double beta  = 0.0;
    vector<double> weights;

    for(int i=0;i<num_particles;i++)
        weights.push_back(particles[i].weight);
																			
    double max_w = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> uni_dist(0.0, max_w);

    int index = dist_distribution(gen);
    vector<Particle> p;

    for(int i=0;i<num_particles;i++){

        beta = beta + uni_dist(gen) * 2.0;
		while(weights[index]<beta){
            beta  = beta - weights[index];
            index = (index + 1) % num_particles;
        }
        p.push_back(particles[index]);
    }
    particles = p;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
