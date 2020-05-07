#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <cudarand.h>
#include <cudarand_kernel.h>
#include <nlohmann/json.hpp>

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

void initialize(std::vector<Person> people, std::vector<Location> places, int numPeople, int numPlaces) {
	Location *loc_ptr;
	for(int i = 0; i < numPlaces; i++) {
		Location loc;
		loc_ptr = &loc;
		loc_ptr->interaction_level = 1.;
		places.push_back(*loc_ptr);
		//TODO: do something with duration function (inheritance?)
	}

	Person *person_ptr;
	for(int i = 0; i < numPeople; i++) {
		Person person;
		person_ptr = &person;
		person_ptr->infection_status = SUSCEPTIBLE;
		people.push_back(*person_ptr);
		places[rand() % places.size()].people_next_step.push_back(person_ptr);
	}

	for(int i = 0; i < numPlaces; i++) {
		std::clog << "Location " << i << " has " << places[i].people_next_step.size() << " people." << std::endl;
	}
}

void updateLocations(std::vector<Location> places, int n) {
	Location* loc_ptr;
	for (int loc_idx = 0; loc_idx < places.size(); loc_idx++) {
		loc_ptr = &places[loc_idx];
		loc_ptr->people.swap(loc_ptr->people_next_step);
		loc_ptr->people_next_step.clear();
	}
}

__global__ void spreadDisease(Location** dev_places, int num_places, int* place_num_people, placeDisease* dev_disease, unsigned long rand_seed) {
	Location* loc_ptr;
	Person* person_ptr;
	loc_ptr = &dev_places[blockIdx.x];
	int num_people = place_num_people[blockIdx.x];

	curandState_t state;
	cudarand_init(rand_seed, blockIdx.x*blockDim.x+threadIdx.x, 0, &state); 

	//determine spread of infection from infected to healthy
	__shared__ int has_sick = 0;
	
	for(int i = 0; i < num_people/blockDim.x; i++){
		int personIdx = i*blockDim.x + threadIdx.x;
		if(personIdx < num_people){
			person_ptr = loc_ptr->people[personIdx];
			// concurrency issue but only care if it 'ever' gets set to 1
			if ((person_ptr->infection_status == SICK) || (person_ptr->infection_status == CARRIER)) {
				num_sick = 1;
			}
		}
	}

	__syncthreads();

	// Propogate infections in places with infected people
	if(has_sick > 0) {
		for(int i = 0; i < num_people/blockDim.x; i++){
			int personIdx = i*blockDim.x + threadIdx.x;
			if(personIdx < num_people){
				person_ptr = loc_ptr->people[personIdx];

				// TODO: scale infection probability properly
				float infection_probability = disease->SPREAD_FACTOR * loc_ptr->interaction_level;
				float r = curand_unifrom(&state);
				if (r < infection_probability) {
					person_ptr->infection_status = CARRIER;
				}
			}
		}
	}
}

void findNextLocations(std::vector<Location> places, std::vector<Person> people, int numPlaces, int numPeople) {
	Location* loc_ptr;
	Person* person_ptr;
	for(int person_idx = 0; person_idx < numPeople; person_idx++){
		person_ptr = &people[person_idx];
		float r = (float) rand() / RAND_MAX;
		if(r < MOVEMENT_PROBABILITY) {
			int new_loc = rand() % numPlaces;
			places[new_loc].people_next_step.push_back( person_ptr );
		} else {
			loc_ptr->people_next_step.push_back( person_ptr );
		}
	}
}

__global__ void advanceInfection(std::vector<Location> places, int num_places, int* place_num_people, Disease* disease, int* susceptible, int* infected, int* recovered, int* deceased, unsigned long rand_seed){
	Location* loc_ptr;
	Person* person_ptr;
	loc_ptr = &dev_places[blockIdx.x];
	int num_people = place_num_people[blockIdx.x];

	curandState_t state;
	cudarand_init(rand_seed, blockIdx.x*blockDim.x+threadIdx.x, 0, &state); 

	for(int i = 0; i < num_people/blockDim.x; i++){
		int personIdx = i*blockDim.x + threadIdx.x;
		if(personIdx < num_people){
			person_ptr = loc_ptr->people[personIdx];
			switch (person_ptr->infection_status) {
				case CARRIER:
					// TODO: Normal Distribution around average times
					if (person_ptr->state_count > (int) disease->AVERAGE_INCUBATION_DURATION) {
						person_ptr->infection_status = SICK;
						person_ptr->state_count = 0;

						// TODO: death rate based on age
						float r = curand_normal(&state);
						person_ptr->to_die = (r < disease->DEATH_RATE);
					} else {
						person_ptr->state_count++;
					}
					break;

				case SICK:
					if (person_ptr->to_die) {
						if (person_ptr->state_count > disease->AVERAGE_TIME_DEATH)
							person_ptr->infection_status = DECEASED;
					} else {
						if (person_ptr->state_count > disease->AVERAGE_TIME_RECOVERY)
							person_ptr->infection_status = RECOVERED;
					}
					person_ptr->state_count++;
					break;
				default:
					break;
			}
		}
	}
}


void collectStatistics(std::vector<Location> places, int n, Disease* disease, int* susceptible, int* infected, int* recovered, int* deceased, unsigned long rand_seed){
	Location* loc_ptr;
	Person* person_ptr;
	loc_ptr = &dev_places[blockIdx.x];
	person_ptr = loc_ptr->people[threadIdx.x];

	switch (person_ptr->infection_status) {
		case SUSCEPTIBLE:
			(*susceptible)++;
			person_ptr->state_count = 0;
			break;
		case CARRIER:
			(*infected)++;

			// TODO: Normal Distribution around average times
			if (person_ptr->state_count > (int) disease->AVERAGE_INCUBATION_DURATION) {
				person_ptr->infection_status = SICK;
				person_ptr->state_count = 0;

				// TODO: death rate based on age
				float r = (float) rand() / RAND_MAX;
				if (r < disease->DEATH_RATE)
					person_ptr->to_die = true;
				else
					person_ptr->to_die = false;
			} else {
				person_ptr->state_count++;
			}
			break;
		case SICK:
			(*infected)++;

			if (person_ptr->to_die) {
				if (person_ptr->state_count > disease->AVERAGE_TIME_DEATH)
					person_ptr->infection_status = DECEASED;
			} else {
				if (person_ptr->state_count > disease->AVERAGE_TIME_RECOVERY)
					person_ptr->infection_status = RECOVERED;
			}
			person_ptr->state_count++;
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
	DEBUG = input_json.value("debug", 0);

	srand(time(NULL));
	
	// All other references to these objects should be pointers or arrays of pointers
	std::vector<Location> places;
	std::vector<Person> people;

	initialize(people, places, pop_size, num_locs);
	
	// Configure disease based on input argument
	json disease_json = input_json.value("disease", input_json);
	Disease disease(disease_json);

	int num_infected = input_json.value("initial_infected", 0);
	int person_to_infect;
	int location_to_infect;

	// Infect initial population.
	// note: people in less populated locations are more likely to be infected.
	// This should only matter for initially infecting large (>25%?) amounts of the population.
	for(int i = 0; i < num_infected; i++) {
		do {
			location_to_infect = rand() % num_locs;
			person_to_infect = rand() % places[location_to_infect].people_next_step.size();
		} while(people[person_to_infect].infection_status != SUSCEPTIBLE);
		people[person_to_infect].infection_status = CARRIER;
		if(DEBUG) {
			std::clog << location_to_infect << " has an infected person" << std::endl;
		}
	}

	// Susciptible/Infected/Recovered/Deceased
	int num_susceptible = pop_size - num_infected;
	int num_recovered = 0;
	int num_deceased = 0;

	Location *loc_ptr;
	Person *person_ptr;

	if(DEBUG) std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	for(int hour = 0; num_infected > 0 && hour < SIMULATION_LENGTH; hour++) {
		updateLocations(places, places.size());
		spreadDisease(places, places.size(), &disease);
		findNextLocations(places, people, places.size(), people.size());
		num_infected = num_susceptible = num_recovered = num_deceased = 0;
		collectStatistics(places, places.size(), &disease, &num_susceptible, &num_infected, &num_recovered, &num_deceased);
		if(DEBUG) std::cout << num_susceptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;
	}
}
