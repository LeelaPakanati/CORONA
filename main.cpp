//----------------------------------
// File: main.cpp
// Author: Leela Pakanati, Taylor Chatfield
// Class: ECE497 Cuda Programming
// Assignment: Final Project
// Purpose: Single-Threaded Simulation Implementation
// Date: 5/25/2020
//----------------------------------

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstring>

#include "./datatypes/location.cpp"
#include "./datatypes/person.cpp"
#include "./datatypes/disease.cpp"
#include "./string_code.cpp"

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

	for(int i = 0; i < numPlaces; i++) {
		std::clog << "Location " << i << " has " << places[i].num_people_next_step << " people." << std::endl;
	}
}

void updateLocations(Location *places, int num_places) {
	//Person temp_people[MAX_LOCATION_CAPACITY];
	for (int loc_idx = 0; loc_idx < num_places; loc_idx++) {
		//memcpy(temp_people, places[loc_idx].people, places[loc_idx].num_people * sizeof(Person));
		memcpy(places[loc_idx].people, places[loc_idx].people_next_step, places[loc_idx].num_people_next_step * sizeof(Person));
		//memcpy(places[loc_idx].people_next_step, temp_people, places[loc_idx].num_people * sizeof(Person));
		places[loc_idx].num_people = places[loc_idx].num_people_next_step;
		places[loc_idx].num_people_next_step = 0;
	}
}

void spreadDisease(Location *places, int num_places, Disease disease) {
	for (int loc_idx = 0; loc_idx < num_places; loc_idx++) {

		//determine spread of infection from infected to healthy
		int num_sick = 0;
		
		// Get number of sick people
		for (int person_idx = 0; person_idx < places[loc_idx].num_people; person_idx++) {
			if (places[loc_idx].people[person_idx].infection_status == SICK || places[loc_idx].people[person_idx].infection_status == CARRIER) {
				num_sick++;
			}
		}
		
		// Propogate infections in places with infected people
		if(num_sick > 0) {
			for (int person_idx = 0; person_idx < places[loc_idx].num_people; person_idx++) {
				if (places[loc_idx].people[person_idx].infection_status == SUSCEPTIBLE) {
					// TODO: scale infection probability properly
					float infection_probability = disease.SPREAD_FACTOR * places[loc_idx].interaction_level;
					float r = (float) rand() / RAND_MAX;
					if (r < infection_probability) {
						places[loc_idx].people[person_idx].infection_status = CARRIER;
					}
				}
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

void collectStatistics(Location *places, int numPlaces, Disease disease, int* susceptible, int* infected, int* recovered, int* deceased) {
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
					places[loc_idx].people[person_idx].state_count = 0;
					break;
				case CARRIER:
					(*infected)++;

					// TODO: Normal Distribution around average times
					if (places[loc_idx].people[person_idx].state_count > (int) disease.AVERAGE_INCUBATION_DURATION) {
						places[loc_idx].people[person_idx].infection_status = SICK;
						places[loc_idx].people[person_idx].state_count = 0;

						// TODO: death rate based on age
						float r = (float) rand() / RAND_MAX;
						if (r < disease.DEATH_RATE)
							places[loc_idx].people[person_idx].to_die = true;
						else
							places[loc_idx].people[person_idx].to_die = false;
					} else {
						places[loc_idx].people[person_idx].state_count++;
					}
					break;
				case SICK:
					(*infected)++;

					if (places[loc_idx].people[person_idx].to_die) {
						if (places[loc_idx].people[person_idx].state_count > disease.AVERAGE_TIME_DEATH)
							places[loc_idx].people[person_idx].infection_status = DECEASED;
					} else {
						if (places[loc_idx].people[person_idx].state_count > disease.AVERAGE_TIME_RECOVERY)
							places[loc_idx].people[person_idx].infection_status = RECOVERED;
					}
					places[loc_idx].people[person_idx].state_count++;
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
	srand(time(NULL));

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
	Location *places = (Location*) malloc(num_locs * sizeof(Location));

	initialize(places, pop_size, num_locs);
	
	int person_to_infect;
	int location_to_infect;

	// Infect initial population.
	// note: people in less populated locations are more likely to be infected.
	// This should only matter for initially infecting large (>25%?) amounts of the population.
	for(int i = 0; i < num_infected; i++) {
		do {
			location_to_infect = rand() % num_locs;
			person_to_infect = rand() % places[location_to_infect].num_people_next_step;
		} while(places[location_to_infect].people_next_step[person_to_infect].infection_status != SUSCEPTIBLE);
		places[location_to_infect].people_next_step[person_to_infect].infection_status = CARRIER;
		if(DEBUG) {
			std::clog << location_to_infect << " has an infected person" << std::endl;
		}
	}

	// Susciptible/Infected/Recovered/Deceased
	int num_susceptible, num_recovered, num_deceased;

	if(DEBUG) std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	for(int hour = 0; num_infected > 0 && hour < SIMULATION_LENGTH; hour++) {
		updateLocations(places, num_locs);
		collectStatistics(places, num_locs, disease, &num_susceptible, &num_infected, &num_recovered, &num_deceased);
		spreadDisease(places, num_locs, disease);
		findNextLocations(places, num_locs);
		if(DEBUG) std::cout << num_susceptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;
	}

	free(places);
}
