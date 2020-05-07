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

// temporary
#define MOVEMENT_PROBABILITY .1

// variables
int SIMULATION_LENGTH = 365*24;
bool DEBUG;

// for convenience
using json = nlohmann::json;

void initialize(Location *places, int numPeople, int numPlaces, int maxSize) {

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

	for(int i = 0; i < numPlaces; i++) {
		std::clog << "Location " << i << " has " << places[i].num_people_next_step << " people." << std::endl;
	}
}

void updateLocations(Location *places, int num_places) {
	Person temp_people[MAX_LOCATION_CAPACITY];
	for (int loc_idx = 0; loc_idx < num_places; loc_idx++) {
		memcpy(temp_people, places[loc_idx].people, places[loc_idx].num_people * sizeof(Person));
		memcpy(places[loc_idx].people, places[loc_idx].people_next_step, places[loc_idx].num_people_next_step * sizeof(Person));
		memcpy(places[loc_idx].people_next_step, temp_people, places[loc_idx].num_people * sizeof(Person));
		places[loc_idx].num_people = places[loc_idx].num_people_next_step;
		places[loc_idx].num_people_next_step = 0;
	}
}

__global__ void spreadDisease(Location* dev_places, int max_size,  Disease disease, unsigned long rand_seed) {
	int loc_idx = blockIdx.x;
	int person_idx;

	curandState_t state;
	curand_init(rand_seed, blockIdx.x*blockDim.x+threadIdx.x, 0, &state); 

	//determine spread of infection from infected to healthy
	__shared__ int has_sick;
	has_sick = 0;
	
	for(int i = 0; i < dev_places[loc_idx].num_people/blockDim.x+1; i++){
		person_idx = i*blockDim.x + threadIdx.x;

		if(person_idx < dev_places[loc_idx].num_people){															// Minimal control divergence
			// concurrency issue but only care if it 'ever' gets set to 1
			if ((dev_places[loc_idx].people[person_idx].infection_status == SICK) || (dev_places[loc_idx].people[person_idx].infection_status == CARRIER)) {			// A lot of control divergence
				has_sick = 1;
			}
		}
	}

	__syncthreads();

	// Propogate infections in places with infected people
	if(has_sick > 0) {
		for(int i = 0; i < dev_places[loc_idx].num_people/blockDim.x+1; i++){
			person_idx = i*blockDim.x + threadIdx.x;
			if(person_idx < dev_places[loc_idx].num_people){															// Minimal control divergence
				if(dev_places[loc_idx].people[person_idx].infection_status == SUSCEPTIBLE){										// A lot of control divergence
					float infection_probability = disease.SPREAD_FACTOR * dev_places[loc_idx].interaction_level;
					float r = curand_uniform(&state);
					if (r < infection_probability) {													// A lot of control divergence
						dev_places[loc_idx].people[person_idx].infection_status = CARRIER;
					}
				}
			}
		}
	}
}

__global__ void advanceInfection(Location* dev_places, int max_size, Disease disease, unsigned long rand_seed){
	int loc_idx = blockIdx.x;
	int person_idx;

	curandState_t state;
	curand_init(rand_seed, blockIdx.x*blockDim.x+threadIdx.x, 0, &state); 

	for(int i = 0; i < dev_places[loc_idx].num_people/blockDim.x+1; i++){
		person_idx = i*blockDim.x + threadIdx.x;
		if(person_idx < dev_places[loc_idx].num_people){															// Minimal control divergence
			switch (dev_places[loc_idx].people[person_idx].infection_status) {
				case CARRIER:
					// TODO: Normal Distribution around average times
					if (dev_places[loc_idx].people[person_idx].state_count > (int) disease.AVERAGE_INCUBATION_DURATION) {
						dev_places[loc_idx].people[person_idx].infection_status = SICK;
						dev_places[loc_idx].people[person_idx].state_count = 0;

						// TODO: death rate based on age
						float r = curand_normal(&state);
						dev_places[loc_idx].people[person_idx].to_die = (r < disease.DEATH_RATE);
					} else {
						dev_places[loc_idx].people[person_idx].state_count++;
					}
					break;

				case SICK:
					if (dev_places[loc_idx].people[person_idx].to_die) {
						if (dev_places[loc_idx].people[person_idx].state_count > disease.AVERAGE_TIME_DEATH)
							dev_places[loc_idx].people[person_idx].infection_status = DECEASED;
					} else {
						if (dev_places[loc_idx].people[person_idx].state_count > disease.AVERAGE_TIME_RECOVERY)
							dev_places[loc_idx].people[person_idx].infection_status = RECOVERED;
					}
					dev_places[loc_idx].people[person_idx].state_count++;
					break;
				default:
					break;
			}
		}
	}
}


void findNextLocations(Location *places, int numPlaces, int maxSize) {
	int new_loc_idx;
	for (int loc_idx = 0; loc_idx < numPlaces; loc_idx++) {
		for (int person_idx = 0; person_idx < places[loc_idx].num_people; person_idx++) {
			float r = (float) rand() / RAND_MAX;
			new_loc_idx = rand() % numPlaces;
			if(r < MOVEMENT_PROBABILITY && places[new_loc_idx].num_people_next_step < maxSize - 1) {
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

	initialize(host_places, pop_size, num_locs, max_size);
	
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

	dim3 dimGrid(num_locs, 1, 1);
	dim3 dimBlock(256, 1, 1);

	int long seed = time(NULL);

	if(DEBUG) std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	for(int hour = 0; num_infected > 0 && hour < SIMULATION_LENGTH; hour++) {
		updateLocations(host_places, num_locs);
		collectStatistics(host_places, num_locs, &num_susceptible, &num_infected, &num_recovered, &num_deceased);

		cudaMemcpy(dev_places, host_places, num_locs * sizeof(struct Location), cudaMemcpyHostToDevice);

		spreadDisease<<<dimGrid, dimBlock>>>(dev_places, max_size, disease, seed);
		advanceInfection<<<dimGrid, dimBlock>>>(dev_places, max_size, disease, seed);

		cudaMemcpy(host_places, dev_places, num_locs * sizeof(struct Location), cudaMemcpyDeviceToHost);

		findNextLocations(host_places, num_locs, max_size);
		if(DEBUG) std::cout << num_susceptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;
	}

	free(host_places);
	cudaFree(dev_places);
}
