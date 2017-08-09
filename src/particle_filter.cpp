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
#include "Eigen/Dense"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 50;
	normal_distribution<double> rand_x(x, std[0]);
	normal_distribution<double> rand_y(y, std[1]);
	normal_distribution<double> rand_theta(theta, std[2]);
    default_random_engine gen;
    for (size_t i = 0; i < num_particles; ++i) {
        Particle sample;
        sample.id = i;
        sample.x = rand_x(gen);
		sample.y = rand_y(gen);
		sample.theta = rand_theta(gen);
        sample.weight = 1;
        particles.push_back(sample);
    }
    weights.resize(num_particles, 1);
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    for (size_t i = 0; i < num_particles; ++i) {
        float x = particles[i].x;
        float y = particles[i].y;
        float theta = particles[i].theta;
        if (fabs(yaw_rate) < 0.01) {
            x += velocity * delta_t * cos(theta);
            y += velocity * delta_t * sin(theta);
            theta = fmod(theta + yaw_rate * delta_t, 2 * M_PI);
        } else {
            x = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
            y = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
            theta = fmod(theta + yaw_rate * delta_t, 2 * M_PI);
        }
        normal_distribution<double> rand_x(x, std_pos[0]);
        normal_distribution<double> rand_y(y, std_pos[1]);
        normal_distribution<double> rand_theta(theta, std_pos[2]);
        particles[i].x = rand_x(gen);
		particles[i].y = rand_y(gen);
		particles[i].theta = rand_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (size_t i = 0; i < observations.size(); ++i) {
		LandmarkObs& obs = observations[i];
		double _min = -1;
		for (size_t j = 0; j < predicted.size(); ++j) {
			double _dist = dist(obs.x, obs.y, predicted[j].x, predicted[j].y);
            if (_min == -1 || _dist < _min) {
				_min = _dist;
				obs.id = j;
			}
		}
	}
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
    std::vector<int> associations;
    std::vector<double> sense_x, sense_y;
    std::vector<LandmarkObs> predicted;

    double sum;
    double denominator = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
    for (size_t i = 0; i < particles.size(); ++i) {
        Particle& p = particles[i];
        for (size_t j = 0; j < map_landmarks.landmark_list.size(); ++j) {
            Map::single_landmark_s& map_landmark = map_landmarks.landmark_list[j];
            float _dist = dist(p.x, p.y, map_landmark.x_f, map_landmark.y_f);
            if (_dist <= sensor_range) {
                associations.push_back(map_landmark.id_i);
                sense_x.push_back(map_landmark.x_f);
                sense_y.push_back(map_landmark.y_f);
                LandmarkObs obs;
                Homotrans(p, map_landmark, &obs);
                cout << "qqq " << obs.x << " " << obs.y << endl;
                predicted.push_back(obs);
            }
        }
        p = SetAssociations(p, associations, sense_x, sense_y);
        dataAssociation(predicted, observations);

		p.weight = 1;
        cout << "kk";
		for (size_t j = 0; j < observations.size(); ++j) {
			LandmarkObs& obs = observations[j];
			double dx = predicted[obs.id].x - obs.x;
			double dy = predicted[obs.id].y - obs.y;
            cout << "x1 " << dx << " " << dy << " " << p.weight;
			p.weight *= exp(-0.5 * (
                pow(dx / std_landmark[0], 2) + pow(dy / std_landmark[1], 2)
            )) / denominator;
            cout << ", " << j << " " << obs.id;
		}
        cout << endl;
		weights[i] = p.weight;
        sum += weights[i];
    }

    cout << "ss";
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = weights[i] / sum;
        cout << " " << weights[i];
    }
    cout << endl;
}

void ParticleFilter::Homotrans(const Particle& p, const Map::single_landmark_s& map_landmark, LandmarkObs* obs)
{
	Eigen::MatrixXd R(3, 3);
	Eigen::MatrixXd T(3, 3);
	R << cos(p.theta), -sin(p.theta), 0,
          sin(p.theta), cos(p.theta), 0,
          0, 0, 1;
    T << 1, 0, p.x,
      0, 1, p.y,
      0, 0 ,1;
	Eigen::MatrixXd O(3, 1);
    O << map_landmark.x_f, map_landmark.y_f, 1;
    Eigen::MatrixXd out = T * R * O;
	obs->id = map_landmark.id_i;
	obs->x = out(0, 0);
	obs->y = out(1, 0);
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles(num_particles);
	default_random_engine gen;
	discrete_distribution<double> dist(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++)
	{
		int idx = dist(gen);
		new_particles[i] = particles[idx];
	}

	particles.swap(new_particles);
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
