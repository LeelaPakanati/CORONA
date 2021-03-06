//----------------------------------
// File: main.cu
// Author: Leela Pakanati, Taylor Chatfield
// Class: ECE497 Cuda Programming
// Assignment: Final Project
// Purpose: Parallelized Simulation Implementation
// Date: 5/25/2020
//----------------------------------

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <cstring>

#include "./datatypes/location.cpp"
#include "./datatypes/person.cpp"
#include "./datatypes/disease.cpp"
#include "./string_code.cpp"
#define BLOCK_WIDTH 256
// temporary
#define MOVEMENT_PROBABILITY .1

// variables
int SIMULATION_LENGTH = 365*24;
bool DEBUG;

void initialize(Location *places, int numPeople, int numPlaces) {

	for(int i = 0; i < numPlaces; i++) {
		places[i].num_people = 0;
		places[i].num_people_next_step = 0;
		places[i].interaction_level = 1.;
	}

	int loc_idx;
	for(int i = 0; i < numPeople; i++) {
		loc_idx = rand() % numPlaces;
		places[loc_idx].people_next_step[places[loc_idx].num_people_next_step].infection_status = SUSCEPTIBLE;
		places[loc_idx].people_next_step[places[loc_idx].num_people_next_step].state_count = 0;
		places[loc_idx].people_next_step[places[loc_idx].num_people_next_step].to_die = 0;
		places[loc_idx].people_next_step[places[loc_idx].num_people_next_step++].id = i;
	}

}

// initialize curand
__global__ void init(unsigned int seed, curandState_t* states){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &states[idx]); 
}

// replace people array with people_next_step
__global__ void updateLocations(Location *places) {
	int loc_idx = blockIdx.x;
	Location* loc_ptr = &places[loc_idx];
	memcpy(loc_ptr->people, loc_ptr->people_next_step, loc_ptr->num_people_next_step * sizeof(Person));
	loc_ptr->num_people = loc_ptr->num_people_next_step;
	loc_ptr->num_people_next_step = 0;
}

// determine if sick people at location and spread disease accordingly
__global__ void spreadDisease(Location* places, Disease disease, curandState_t* states) {
	int loc_idx = blockIdx.x;
	Location* loc_ptr = &places[loc_idx];
	int person_idx;

	//determine spread of infection from infected to healthy
	__shared__ int num_sick;
	if (threadIdx.x == 0){
		num_sick = 0;
	}

	person_idx = threadIdx.x;
	Person* person_ptr = &loc_ptr->people[person_idx];

	if (person_idx < loc_ptr->num_people){															// Minimal control divergence
		atomicAdd(&num_sick, ((person_ptr->infection_status == SICK) || (person_ptr->infection_status == CARRIER)));
	}

	__syncthreads();

	// Propogate infections in places with infected people
	if (num_sick > 0) {
		Person* person_ptr = &loc_ptr->people[person_idx];
		if (person_idx < loc_ptr->num_people){															// Minimal control divergence
			if (person_ptr->infection_status == SUSCEPTIBLE){										// A lot of control divergence
				float infection_probability = disease.SPREAD_FACTOR * loc_ptr->interaction_level;
				float r = curand_uniform(&states[blockIdx.x*blockDim.x+threadIdx.x]);
				if (r < infection_probability) {													// A lot of control divergence
					person_ptr->infection_status = CARRIER;
				}
			}
		}
	}
}

// advance infection status for sick people
__global__ void advanceInfection(Location* places, Disease disease, curandState_t* states){
	int loc_idx = blockIdx.x;
	Location* loc_ptr = &places[loc_idx];
	int person_idx;

	person_idx = threadIdx.x;
	Person* person_ptr = &loc_ptr->people[person_idx];
	if (person_idx < loc_ptr->num_people){															// Minimal control divergence
		switch (person_ptr->infection_status) {										// Massive control divergence
			case CARRIER:
				// TODO: Normal Distribution around average times
				if (person_ptr->state_count > (int) disease.AVERAGE_INCUBATION_DURATION) {
					person_ptr->infection_status = SICK;
					person_ptr->state_count = 0;

					// TODO: death rate based on age
					float r = curand_uniform(&states[blockIdx.x*blockDim.x+threadIdx.x]);
					person_ptr->to_die = (r < disease.DEATH_RATE);
				} else {
					person_ptr->state_count++;
				}
				break;

			case SICK:
				if (person_ptr->to_die) {
					if (person_ptr->state_count > disease.AVERAGE_TIME_DEATH)
						person_ptr->infection_status = DECEASED;
				} else {
					if (person_ptr->state_count > disease.AVERAGE_TIME_RECOVERY)
						person_ptr->infection_status = RECOVERED;
				}
				person_ptr->state_count++;
				break;
			default:
				break;
		}
	}
}

// collect statistics about infected status of people at each location
__global__ void collectStatistics(Location *places, int* susceptible, int* infected, int* recovered, int* deceased) {
	int loc_idx = blockIdx.x;
	Location* loc_ptr = &places[loc_idx];
	int person_idx = threadIdx.x;
	Person* person_ptr = &loc_ptr->people[person_idx];

	__shared__ int num_s;
	__shared__ int num_i;
	__shared__ int num_r;
	__shared__ int num_d;
	
	if(threadIdx.x == 0){
		num_s = 0;
		num_i = 0;
		num_r = 0;
		num_d = 0;
	}

	__syncthreads();

	if (person_idx < loc_ptr->num_people){
		switch (person_ptr->infection_status) {	//massive control divergence
			case SUSCEPTIBLE:
				atomicAdd(&num_s, 1);
				break;
			case CARRIER:
				atomicAdd(&num_i, 1);
			case SICK:
				atomicAdd(&num_i, 1);
				break;
			case RECOVERED:
				atomicAdd(&num_r, 1);
				break;
			case DECEASED:
				atomicAdd(&num_d, 1);
				break;
			default:
				break;
		}
	}

	__syncthreads();

	susceptible[loc_idx] = num_s;
	infected[loc_idx] = num_i;
	recovered[loc_idx] = num_r;
	deceased[loc_idx] = num_d;

}

// add up statistics of all locations
__global__ void addStatistics(int pow2_num_locs, int* susceptible, int* infected, int* recovered, int* deceased, int* stats, int hour) {
	//using reduction to count stats
	for (unsigned int stride = pow2_num_locs/2; stride > 0; stride /= 2){
		__syncthreads();
		// switch over blockIdx so no control divergence
		if (threadIdx.x + stride < blockDim.x){
			switch(blockIdx.x) {
				case 0:
					susceptible[threadIdx.x] += susceptible[threadIdx.x+stride];
					break;
				case 1:
					infected[threadIdx.x] += infected[threadIdx.x+stride];
					break;
				case 2:
					recovered[threadIdx.x] += recovered[threadIdx.x+stride];
					break;
				case 3:
					deceased[threadIdx.x] += deceased[threadIdx.x+stride];
					break;
				default:
					break;
			}
		}
	}

	__syncthreads();

	if(threadIdx.x == 0){
		switch(blockIdx.x) {
			case 0:
				stats[hour*4] += susceptible[0];
				break;
			case 1:
				stats[hour*4+1] += infected[0];
				break;
			case 2:
				stats[hour*4+2] += recovered[0];
				break;
			case 3:
				stats[hour*4+3] += deceased[0];
				break;
			default:
				break;
		}
	}
}



// determine next location for people
__global__ void findNextLocations(Location *places, int numPlaces, curandState_t* states) {
	int loc_idx = blockIdx.x;
	Location* loc_ptr = &places[loc_idx];
	int person_idx = threadIdx.x;
	Person* person_ptr = &loc_ptr->people[person_idx];

	__shared__ Person local_people_next_step[256];
	__shared__ int local_num_people_next_step;
	if(threadIdx.x == 0){
		local_num_people_next_step = 0;
	}
	__syncthreads();

	if (person_idx < loc_ptr->num_people){
		float r = curand_uniform(&states[blockIdx.x*blockDim.x+threadIdx.x]);
		int new_loc_idx = (int) (curand_uniform(&states[blockIdx.x*blockDim.x+threadIdx.x]) * numPlaces);

		// if probability matches to go to new location && new location isn't full; or if current location is full, go to new loc
		if ((r < MOVEMENT_PROBABILITY && places[new_loc_idx].num_people_next_step < MAX_LOCATION_CAPACITY - 1)
				|| (local_num_people_next_step + places[new_loc_idx].num_people_next_step > MAX_LOCATION_CAPACITY) ){

			int person_new_idx = atomicAdd(&places[new_loc_idx].num_people_next_step, 1);
			memcpy(&places[new_loc_idx].people_next_step[person_new_idx], person_ptr, sizeof(Person));
		} else {
			int person_new_idx = atomicAdd(&local_num_people_next_step, 1);
			memcpy(&local_people_next_step[person_new_idx], person_ptr, sizeof(Person));
		}
	}

	__syncthreads();

	// combine global and local copies of next people arrays for location
	if(threadIdx.x < local_num_people_next_step){
		memcpy(&places[loc_idx].people_next_step[places[loc_idx].num_people_next_step + threadIdx.x], &local_people_next_step[threadIdx.x], sizeof(Person));	
	}

	if(threadIdx.x == 0)
		places[loc_idx].num_people_next_step += local_num_people_next_step;

}

int main(int argc, char** argv){

	// Parse arguments
	if (argc < 2){
		std::cerr << "Usage : " << argv[0] << " <input file>" << std::endl;
		return 0;
	}

	// Parse argument file
	std::string input_file_name = argv[1];
	std::clog << "Reading file " << input_file_name << " for starting conditions" << std::endl;

	// TODO: Add  more complex person/location config
	std::ifstream input_file(input_file_name);
	std::string myText;
	int pop_size = 0;
	int num_locs = 0;
	int num_infected = 0;
	Disease disease;

	while (getline(input_file, myText)){
		//std::cout << myText << std::endl;
		std::string token = myText.substr(0, myText.find(":"));
		std::string value = myText.substr(myText.find(":") + 1);

		switch (hash_it(token)) {
			case e_debug:
				DEBUG = atoi(value.c_str());
				break;
			case e_population_size:
				pop_size = atoi(value.c_str());
				break;
			case e_num_locations:
				num_locs = atoi(value.c_str());
				break;
			case e_initial_infected:
				num_infected = atoi(value.c_str());
				break;
			case e_SPREAD_FACTOR:
				disease.SPREAD_FACTOR = atof(value.c_str());
				break;
			case e_CARRIER_PROBABILITY:
				disease.CARRIER_PROBABILITY = atof(value.c_str());
				break;
			case e_AVERAGE_INCUBATION_DURATION:
				disease.AVERAGE_INCUBATION_DURATION = atof(value.c_str());
				break;
			case e_AVERAGE_TIME_DEATH:
				disease.AVERAGE_TIME_DEATH = atof(value.c_str());
				break;
			case e_AVERAGE_TIME_RECOVERY:
				disease.AVERAGE_TIME_RECOVERY = atof(value.c_str());
				break;
			case e_DEATH_RATE:
				disease.DEATH_RATE = atof(value.c_str());
				break;
			default:
				std::cout << "Invalid sample file entry: " << token << std::endl;
		}
	}

	// Setup Cuda Rand
	curandState_t* states;
	cudaMalloc((void**) &states, num_locs * BLOCK_WIDTH * sizeof(curandState_t));

	init<<<num_locs, BLOCK_WIDTH>>>(time(NULL), states);

	// All other references to these objects should be pointers or arrays of pointers
	Location* host_places; //= (Location*) malloc(num_locs * sizeof(Location));
	Location* dev_places;
	cudaHostAlloc((void **) &host_places, num_locs * sizeof(struct Location), cudaHostAllocDefault);
	cudaMalloc((void **) &dev_places, num_locs * sizeof(struct Location));

	initialize(host_places, pop_size, num_locs);

	int person_to_infect;
	int location_to_infect;

	// Infect initial population.
	// note: people in less populated locations are more likely to be infected.
	// This should only matter for initially infecting large (>25%?) amounts of the population.
	for(int i = 0; i < num_infected; i++) {
		do {
			location_to_infect = rand() % num_locs;
			person_to_infect = rand() % host_places[location_to_infect].num_people_next_step;
		} while(host_places[location_to_infect].people_next_step[person_to_infect].infection_status != SUSCEPTIBLE);
		host_places[location_to_infect].people_next_step[person_to_infect].infection_status = CARRIER;
		if (DEBUG) {
			std::clog << location_to_infect << " has an infected person" << std::endl;
		}
	}

	// Susciptible/Infected/Recovered/Deceased
	int *d_num_susceptible, *d_num_recovered, *d_num_deceased, *d_num_infected;
	cudaMalloc((void**) &d_num_susceptible, num_locs*sizeof(int));
	cudaMalloc((void**) &d_num_infected, num_locs*sizeof(int));
	cudaMalloc((void**) &d_num_recovered, num_locs*sizeof(int));
	cudaMalloc((void**) &d_num_deceased, num_locs*sizeof(int));

	int pow2_num_locs = pow(2, ceil(log2(num_locs)));

	int* stats = (int*) malloc(SIMULATION_LENGTH * 4 * sizeof(int));
	int* dev_stats;
	cudaMalloc((void**) &dev_stats, SIMULATION_LENGTH * 4 * sizeof(int));

	if(DEBUG) std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	cudaMemcpy(dev_places, host_places, num_locs * sizeof(struct Location), cudaMemcpyHostToDevice);
	int hour = 0;
	for(hour = 0; num_infected > 0 && hour < SIMULATION_LENGTH; hour++) {

		updateLocations<<<num_locs, 1>>>(dev_places);

		spreadDisease<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, states);
		advanceInfection<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, states);
		collectStatistics<<<num_locs, BLOCK_WIDTH>>>(dev_places, d_num_susceptible, d_num_infected, d_num_recovered, d_num_deceased);
		addStatistics<<<4, num_locs>>>(pow2_num_locs, d_num_susceptible, d_num_infected, d_num_recovered, d_num_deceased, dev_stats, hour);

		cudaMemcpy(&num_infected, &d_num_infected[0], sizeof(int), cudaMemcpyDeviceToHost);

		findNextLocations<<<num_locs, BLOCK_WIDTH>>>(dev_places, num_locs, states);
	}

	cudaMemcpy(stats, dev_stats, hour*4*sizeof(int), cudaMemcpyDeviceToHost);

	if (DEBUG)
		for(int i = 0; i < hour; i++)
			std::cout << stats[i*4] << "," << stats[i*4+1] << "," << stats[i*4+2] << "," << stats[i*4+3] << std::endl;


	//free(host_places);
	cudaFreeHost(host_places);
	cudaFree(dev_places);
}
