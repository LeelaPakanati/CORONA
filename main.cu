#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <nlohmann/json.hpp>
#include <curand.h>
#include <curand_kernel.h>

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

__global__ void updateLocations(Location *places) {
	int loc_idx = blockIdx.x;
	//memcpy(temp_people, places[loc_idx].people, places[loc_idx].num_people * sizeof(Person));
	memcpy(places[loc_idx].people, places[loc_idx].people_next_step, places[loc_idx].num_people_next_step * sizeof(Person));
	//memcpy(places[loc_idx].people_next_step, temp_people, places[loc_idx].num_people * sizeof(Person));
	places[loc_idx].num_people = places[loc_idx].num_people_next_step;
	places[loc_idx].num_people_next_step = 0;
}

__global__ void spreadDisease(Location* places, Disease disease, unsigned long rand_seed) {
	int loc_idx = blockIdx.x;
	int person_idx;

	curandState_t state;
	curand_init(rand_seed, blockIdx.x*blockDim.x+threadIdx.x, 0, &state); 

	//determine spread of infection from infected to healthy
	__shared__ bool has_sick[BLOCK_WIDTH];
	has_sick[threadIdx.x] = false;

	__syncthreads();
	
	for(int i = 0; i < places[loc_idx].num_people/blockDim.x+1; i++){
		person_idx = i*blockDim.x + threadIdx.x;

		if(person_idx < places[loc_idx].num_people){															// Minimal control divergence
			if ((places[loc_idx].people[person_idx].infection_status == SICK) || (places[loc_idx].people[person_idx].infection_status == CARRIER)) {			// A lot of control divergence
				has_sick[threadIdx.x] = true;
			}
		}
	}

	__syncthreads();
	
	bool spread = false;
	for(int i = 0; i < BLOCK_WIDTH; i++)
		if(has_sick[i])
			spread = true;

	__syncthreads();

	// Propogate infections in places with infected people
	if(spread) {
		for(int i = 0; i < places[loc_idx].num_people/blockDim.x+1; i++){
			person_idx = i*blockDim.x + threadIdx.x;
			if(person_idx < places[loc_idx].num_people){															// Minimal control divergence
				if(places[loc_idx].people[person_idx].infection_status == SUSCEPTIBLE){										// A lot of control divergence
					float infection_probability = disease.SPREAD_FACTOR * places[loc_idx].interaction_level;
					float r = curand_uniform(&state);
					if (r < infection_probability) {													// A lot of control divergence
						places[loc_idx].people[person_idx].infection_status = CARRIER;
					}
				}
			}
		}
	}
}

__global__ void advanceInfection(Location* places, Disease disease, unsigned long rand_seed){
	int loc_idx = blockIdx.x;
	int person_idx;

	curandState_t state;
	curand_init(rand_seed, blockIdx.x*blockDim.x+threadIdx.x, 0, &state); 

	for(int i = 0; i < places[loc_idx].num_people/blockDim.x+1; i++){
		person_idx = i*blockDim.x + threadIdx.x;
		if(person_idx < places[loc_idx].num_people){															// Minimal control divergence
			switch (places[loc_idx].people[person_idx].infection_status) {
				case CARRIER:
					// TODO: Normal Distribution around average times
					if (places[loc_idx].people[person_idx].state_count > (int) disease.AVERAGE_INCUBATION_DURATION) {
						places[loc_idx].people[person_idx].infection_status = SICK;
						places[loc_idx].people[person_idx].state_count = 0;

						// TODO: death rate based on age
						float r = curand_uniform(&state);
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

void collectStatistics(Location *places, int numPlaces, int* susceptible, int* infected, int* recovered, int* deceased) {
	(*susceptible) = 0;
	(*infected) = 0;
	(*recovered) = 0;
	(*deceased) = 0;
	for (int loc_idx = 0; loc_idx < numPlaces; loc_idx++) {
		// Get number of sick people and set of susceptible people
		for (int person_idx = 0; person_idx < places[loc_idx].num_people; person_idx++) {
			switch (places[loc_idx].people[person_idx].infection_status) {
				case SUSCEPTIBLE:
					(*susceptible)++;
					break;
				case CARRIER:
					(*infected)++;
					break;
				case SICK:
					(*infected)++;
					break;
				case RECOVERED:
					(*recovered)++;
					break;
				case DECEASED:
					(*deceased)++;
					break;
				default:
					break;
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

	int long seed = time(NULL);

	if(DEBUG) std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	for(int hour = 0; num_infected > 0 && hour < SIMULATION_LENGTH; hour++) {
		cudaMemcpy(dev_places, host_places, num_locs * sizeof(struct Location), cudaMemcpyHostToDevice);

		updateLocations<<<num_locs, 1>>>(dev_places);
		spreadDisease<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, seed);
		advanceInfection<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, seed);

		cudaMemcpy(host_places, dev_places, num_locs * sizeof(struct Location), cudaMemcpyDeviceToHost);

		collectStatistics(host_places, num_locs, &num_susceptible, &num_infected, &num_recovered, &num_deceased);
		findNextLocations(host_places, num_locs);
		if(DEBUG) std::cout << num_susceptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;
	}

	free(host_places);
	cudaFree(dev_places);
}
