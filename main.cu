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
	Location* loc_ptr = &places[loc_idx];
	//memcpy(temp_people, places[loc_idx].people, places[loc_idx].num_people * sizeof(Person));
	//if (threadIdx.x == 0){
	memcpy(loc_ptr->people, loc_ptr->people_next_step, loc_ptr->num_people_next_step * sizeof(Person));
	//memcpy(places[loc_idx].people_next_step, temp_people, places[loc_idx].num_people * sizeof(Person));
	loc_ptr->num_people = loc_ptr->num_people_next_step;
	loc_ptr->num_people_next_step = 0;
	//}
}

__global__ void spreadDisease(Location* places, Disease disease, unsigned long rand_seed) {
	int loc_idx = blockIdx.x;
	Location* loc_ptr = &places[loc_idx];
	int person_idx;

	curandState_t state;
	curand_init(rand_seed, blockIdx.x*blockDim.x+threadIdx.x, 0, &state); 

	//determine spread of infection from infected to healthy
	__shared__ int has_sick[BLOCK_WIDTH];
	has_sick[threadIdx.x] = 0;

	//for(int i = 0; i < loc_ptr->num_people/blockDim.x+1; i++){
	//	person_idx = i*blockDim.x + threadIdx.x;
	person_idx = threadIdx.x;
	Person* person_ptr = &loc_ptr->people[person_idx];

	if (person_idx < loc_ptr->num_people){															// Minimal control divergence
		has_sick[threadIdx.x] =  ((person_ptr->infection_status == SICK) || (person_ptr->infection_status == CARRIER)); 
	}
	//}

	__syncthreads();

	//using reduction to add number of sick people
	for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
		if (threadIdx.x < stride)
			has_sick[threadIdx.x] += has_sick[threadIdx.x+stride];

	// Propogate infections in places with infected people
	if (has_sick[0] > 0) {
		//for(int i = 0; i < loc_ptr->num_people/blockDim.x+1; i++){
		//	person_idx = i*blockDim.x + threadIdx.x;
		Person* person_ptr = &loc_ptr->people[person_idx];
		if (person_idx < loc_ptr->num_people){															// Minimal control divergence
			if (person_ptr->infection_status == SUSCEPTIBLE){										// A lot of control divergence
				float infection_probability = disease.SPREAD_FACTOR * loc_ptr->interaction_level;
				float r = curand_uniform(&state);
				if (r < infection_probability) {													// A lot of control divergence
					person_ptr->infection_status = CARRIER;
				}
			}
		}
		//}
	}
}

__global__ void advanceInfection(Location* places, Disease disease, unsigned long rand_seed){
	int loc_idx = blockIdx.x;
	Location* loc_ptr = &places[loc_idx];
	int person_idx;

	curandState_t state;
	curand_init(rand_seed, blockIdx.x*blockDim.x+threadIdx.x, 0, &state); 

	//for(int i = 0; i < loc_ptr->num_people/blockDim.x+1; i++){
	//	person_idx = i*blockDim.x + threadIdx.x;
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
					float r = curand_uniform(&state);
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
		//}
}
}

__global__ void collectStatistics(Location *places, int* susceptible, int* infected, int* recovered, int* deceased) {
	int loc_idx = blockIdx.x;
	Location* loc_ptr = &places[loc_idx];
	int person_idx = threadIdx.x;
	Person* person_ptr = &loc_ptr->people[person_idx];
	int idx = loc_idx * MAX_LOCATION_CAPACITY + person_idx;
	susceptible[idx] = 0;
	infected[idx] = 0;
	recovered[idx] = 0;
	deceased[idx] = 0;
	if (person_idx < loc_ptr->num_people){
		switch (person_ptr->infection_status) {	//massive control divergence
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

//__global__ void simulationKernel(Location *places, Disease disease, unsigned long rand_seed, int* susceptible, int* infected, int* recovered, int* deceased) {
//	updateLocations(places);
//	spreadDisease(places, disease, rand_seed);
//	advanceInfection(places, disease, rand_seed);
//	collectStatistics(places, susceptible, infected, recovered, deceased);
//}

void findNextLocations(Location *places, int numPlaces) {
	int new_loc_idx;
	for (int loc_idx = 0; loc_idx < numPlaces; loc_idx++) {
		for (int person_idx = 0; person_idx < places[loc_idx].num_people; person_idx++) {
			float r = (float) rand() / RAND_MAX;
			new_loc_idx = rand() % numPlaces;
			if (r < MOVEMENT_PROBABILITY && places[new_loc_idx].num_people_next_step < MAX_LOCATION_CAPACITY - 1) {
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
	std::string myText;
	int pop_size = 0;
	int num_locs = 0;
	int max_size = 0;
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
			case e_max_size:
				max_size = atoi(value.c_str());
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

	// All other references to these objects should be pointers or arrays of pointers
	Location* host_places = (Location*) malloc(num_locs * sizeof(Location));
	Location* dev_places;
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
	int num_susceptible, num_recovered, num_deceased;
	int *d_num_susceptible, *d_num_recovered, *d_num_deceased, *d_num_infected;
	cudaMalloc((void**) &d_num_susceptible, num_locs*MAX_LOCATION_CAPACITY*sizeof(int));
	cudaMalloc((void**) &d_num_infected, num_locs*MAX_LOCATION_CAPACITY*sizeof(int));
	cudaMalloc((void**) &d_num_recovered, num_locs*MAX_LOCATION_CAPACITY*sizeof(int));
	cudaMalloc((void**) &d_num_deceased, num_locs*MAX_LOCATION_CAPACITY*sizeof(int));

	int long seed = time(NULL);

	if (DEBUG) std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	for(int hour = 0; num_infected > 0 && hour < SIMULATION_LENGTH; hour++) {
		cudaMemcpy(dev_places, host_places, num_locs * sizeof(struct Location), cudaMemcpyHostToDevice);

		updateLocations<<<num_locs, 1>>>(dev_places);
		spreadDisease<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, seed);
		advanceInfection<<<num_locs, BLOCK_WIDTH>>>(dev_places, disease, seed);
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
		if (DEBUG) std::cout << num_susceptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;
	}

	free(host_places);
	cudaFree(dev_places);
}
