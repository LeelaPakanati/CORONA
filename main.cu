#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <nlohmann/json.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "./datatypes/location.cpp"
#include "./datatypes/person.cpp"
#include "./datatypes/disease.cpp"

#define BLOCK_WIDTH 256
// temporary
#define MOVEMENT_PROBABILITY .1

// variables
int SIMULATION_LENGTH = 365*24;
bool DEBUG;

// for convenience
using json = nlohmann::json;

void initialize(Location *places, int numPeople, int numPlaces) {

	for(int i = 0; i < numPlaces; i++) {
		places[i].num_people = 0;
		places[i].num_people_next_step = 0;
		places[i].interaction_level = 1.;
	}

	int loc_idx;
	for(int i = 0; i < numPeople; i++) {
		//make sure within max size
		loc_idx = rand() % numPlaces;
		places[loc_idx].people_next_step[places[loc_idx].num_people_next_step].infection_status = SUSCEPTIBLE;
		places[loc_idx].people_next_step[places[loc_idx].num_people_next_step].state_count = 0;
		places[loc_idx].people_next_step[places[loc_idx].num_people_next_step].to_die = 0;
		places[loc_idx].people_next_step[places[loc_idx].num_people_next_step++].id = i;
	}

	for(int i = 1; i < numPlaces; i++) {
		std::clog << "Location " << i << " has " << places[i].num_people_next_step << " people." << std::endl;
	}
}

__global__ void init(unsigned int seed, curandState_t* states){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &states[idx]); 
}

__global__ void updateLocations(Location *places) {
	int loc_idx = blockIdx.x;
	//memcpy(temp_people, places[loc_idx].people, places[loc_idx].num_people * sizeof(Person));
	memcpy(places[loc_idx].people, places[loc_idx].people_next_step, places[loc_idx].num_people_next_step * sizeof(Person));
	//memcpy(places[loc_idx].people_next_step, temp_people, places[loc_idx].num_people * sizeof(Person));
	places[loc_idx].num_people = places[loc_idx].num_people_next_step;
	places[loc_idx].num_people_next_step = 0;
}

__global__ void spreadDisease(Location* places, Disease disease, curandState_t* states) {
	int loc_idx = blockIdx.x;
	int person_idx;

	//determine spread of infection from infected to healthy
	__shared__ bool has_sick[BLOCK_WIDTH];
	has_sick[threadIdx.x] = false;
	
	for(int i = 0; i < places[loc_idx].num_people/blockDim.x+1; i++){
		person_idx = i*blockDim.x + threadIdx.x;

		if(person_idx < places[loc_idx].num_people){															// Minimal control divergence
			if ((places[loc_idx].people[person_idx].infection_status == SICK) || (places[loc_idx].people[person_idx].infection_status == CARRIER)) {			// A lot of control divergence
				has_sick[threadIdx.x] = true;
			}
		}
	}

	__syncthreads();
	
	// Inneficient; use reduction?
	bool spread = false;
	for(int i = 0; i < BLOCK_WIDTH; i++)
		if(has_sick[i])
			spread = true;

	// Propogate infections in places with infected people
	if(spread) {
		for(int i = 0; i < places[loc_idx].num_people/blockDim.x+1; i++){
			person_idx = i*blockDim.x + threadIdx.x;
			if(person_idx < places[loc_idx].num_people){															// Minimal control divergence
				if(places[loc_idx].people[person_idx].infection_status == SUSCEPTIBLE){										// A lot of control divergence
					float infection_probability = disease.SPREAD_FACTOR * places[loc_idx].interaction_level;
					float r = curand_uniform(&states[blockIdx.x*blockDim.x+threadIdx.x]);
					if (r < infection_probability) {													// A lot of control divergence
						places[loc_idx].people[person_idx].infection_status = CARRIER;
					}
				}
			}
		}
	}
}

__global__ void advanceInfection(Location* places, Disease disease, curandState_t* states){
	int loc_idx = blockIdx.x;
	int person_idx;


	for(int i = 0; i < places[loc_idx].num_people/blockDim.x+1; i++){
		person_idx = i*blockDim.x + threadIdx.x;
		if(person_idx < places[loc_idx].num_people){															// Minimal control divergence
			switch (places[loc_idx].people[person_idx].infection_status) {										// Massive control divergence
				case CARRIER:
					// TODO: Normal Distribution around average times
					if (places[loc_idx].people[person_idx].state_count > (int) disease.AVERAGE_INCUBATION_DURATION) {
						places[loc_idx].people[person_idx].infection_status = SICK;
						places[loc_idx].people[person_idx].state_count = 0;

						// TODO: death rate based on age
						float r = curand_uniform(&states[blockIdx.x*blockDim.x+threadIdx.x]);
						places[loc_idx].people[person_idx].to_die = (r < disease.DEATH_RATE);
					} else {
						places[loc_idx].people[person_idx].state_count++;
					}
					break;

				case SICK:
					if (places[loc_idx].people[person_idx].to_die) {
						if (places[loc_idx].people[person_idx].state_count > disease.AVERAGE_TIME_DEATH)
							places[loc_idx].people[person_idx].infection_status = DECEASED;
					} else {
						if (places[loc_idx].people[person_idx].state_count > disease.AVERAGE_TIME_RECOVERY)
							places[loc_idx].people[person_idx].infection_status = RECOVERED;
					}
					places[loc_idx].people[person_idx].state_count++;
					break;
				default:
					break;
			}
		}
	}
}

__global__ void collectStatistics(Location *places, int* susceptible, int* infected, int* recovered, int* deceased) {
	int loc_idx = blockIdx.x;
	int person_idx = threadIdx.x;
	int idx = loc_idx * MAX_LOCATION_CAPACITY + person_idx;
	susceptible[idx] = 0;
	infected[idx] = 0;
	recovered[idx] = 0;
	deceased[idx] = 0;
	if(person_idx < places[loc_idx].num_people){
		switch (places[loc_idx].people[person_idx].infection_status) {	//massive control divergence
			case SUSCEPTIBLE:
				susceptible[idx] = 1;
				break;
			case CARRIER:
				infected[idx] = 1;
				break;
			case SICK:
				infected[idx] = 1;
				break;
			case RECOVERED:
				recovered[idx] = 1;
				break;
			case DECEASED:
				deceased[idx] = 1;
				break;
			default:
				break;
		}
	}
}


void findNextLocations(Location *places, int numPlaces) {
	int new_loc_idx;
	for (int loc_idx = 0; loc_idx < numPlaces; loc_idx++) {
		for (int person_idx = 0; person_idx < places[loc_idx].num_people; person_idx++) {
			float r = (float) rand() / RAND_MAX;
			new_loc_idx = rand() % numPlaces;
			if(r < MOVEMENT_PROBABILITY && places[new_loc_idx].num_people_next_step < MAX_LOCATION_CAPACITY - 1) {
				memcpy(&places[new_loc_idx].people_next_step[places[new_loc_idx].num_people_next_step++], &places[loc_idx].people[person_idx], sizeof(Person));
			} else {
				memcpy(&places[loc_idx].people_next_step[places[loc_idx].num_people_next_step++], &places[loc_idx].people[person_idx], sizeof(Person));
			}
		}
	}
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
	json input_json = json::parse(input_file);
	
	int pop_size = input_json.value("population_size", 0);
	int num_locs = input_json.value("num_locations", 0);
	int max_size = input_json.value("max_size", 0);
	DEBUG = input_json.value("debug", 0);
	
	// Setup Cuda Rand
	curandState_t* states;
	cudaMalloc((void**) &states, num_locs * BLOCK_WIDTH * sizeof(curandState_t));

	init<<<num_locs, BLOCK_WIDTH>>>(time(NULL), states);

	// All other references to these objects should be pointers or arrays of pointers
	Location *host_places = (Location*) malloc(num_locs * sizeof(Location));
	Location* dev_places;
	cudaMalloc((void **) &dev_places, num_locs * sizeof(struct Location));

	initialize(host_places, pop_size, num_locs);
	
	// Configure disease based on input argument
	json disease_json = input_json.value("disease", input_json);
	Disease disease;
	disease.SPREAD_FACTOR = disease_json.value("SPREAD_FACTOR", 0.0);
	disease.CARRIER_PROBABILITY = disease_json.value("CARRIER_PROBABILITY", 0.0);
	disease.AVERAGE_INCUBATION_DURATION = disease_json.value("AVERAGE_INCUBATION_DURATION", 0.0);
	disease.AVERAGE_TIME_DEATH = disease_json.value("AVERAGE_TIME_DEATH", 0.0);
	disease.AVERAGE_TIME_RECOVERY = disease_json.value("AVERAGE_TIME_RECOVERY", 0.0);
	disease.DEATH_RATE = disease_json.value("DEATH_RATE", 0.0);

	int num_infected = input_json.value("initial_infected", 0);
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
		if(DEBUG) {
			std::clog << location_to_infect << " has an infected person" << std::endl;
		}
	}

	// Susciptible/Infected/Recovered/Deceased
	int num_susceptible, num_recovered, num_deceased;
	int *d_num_susceptible, *d_num_recovered, *d_num_deceased, *d_num_infected;
	cudaMalloc((void**) &d_num_susceptible, num_locs*MAX_LOCATION_CAPACITY*sizeof(int));
	cudaMalloc((void**) &d_num_infected, num_locs*MAX_LOCATION_CAPACITY*sizeof(int));
	cudaMalloc((void**) &d_num_recovered, num_locs*MAX_LOCATION_CAPACITY*sizeof(int));
	cudaMalloc((void**) &d_num_deceased, num_locs*MAX_LOCATION_CAPACITY*sizeof(int));

	if(DEBUG) std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	for(int hour = 0; num_infected > 0 && hour < SIMULATION_LENGTH; hour++) {
		cudaMemcpy(dev_places, host_places, num_locs * sizeof(struct Location), cudaMemcpyHostToDevice);

		updateLocations<<<num_locs, 1>>>(dev_places);

		spreadDisease<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, states);
		advanceInfection<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, states);
		collectStatistics<<<num_locs, BLOCK_WIDTH>>>(dev_places, d_num_susceptible, d_num_infected, d_num_recovered, d_num_deceased);
		//simulationKernel<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, seed, d_num_susceptible, d_num_infected, d_num_recovered, d_num_deceased);

		thrust::device_ptr<int> d_sus_ptr(d_num_susceptible);
		thrust::device_ptr<int> d_inf_ptr(d_num_infected);
		thrust::device_ptr<int> d_rec_ptr(d_num_recovered);
		thrust::device_ptr<int> d_dec_ptr(d_num_deceased);

		num_susceptible = thrust::reduce(d_sus_ptr, d_sus_ptr + num_locs*MAX_LOCATION_CAPACITY);
		num_infected = thrust::reduce(d_inf_ptr, d_inf_ptr + num_locs*MAX_LOCATION_CAPACITY);
		num_recovered = thrust::reduce(d_rec_ptr, d_rec_ptr + num_locs*MAX_LOCATION_CAPACITY);
		num_deceased = thrust::reduce(d_dec_ptr, d_dec_ptr + num_locs*MAX_LOCATION_CAPACITY);

		cudaMemcpy(host_places, dev_places, num_locs * sizeof(struct Location), cudaMemcpyDeviceToHost);

		findNextLocations(host_places, num_locs);
		if(DEBUG) std::cout << num_susceptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;
	}

	free(host_places);
	cudaFree(dev_places);
}
