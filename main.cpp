#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <nlohmann/json.hpp>

#include "./datatypes/location.cpp"
#include "./datatypes/person.cpp"
#include "./datatypes/disease.cpp"

#define MOVEMENT_PROBABILITY .1

// for convenience
using json = nlohmann::json;

int main(int argc, char** argv){
	// Parse arguments
	if (argc < 2){
		std::cerr << "Usage : " << argv[0] << " <input file>" << std::endl;
		return 0;
	}

	std::string input_file_name = argv[1];
	std::clog << "Reading file " << input_file_name << " for starting conditions" << std::endl;

	// TODO: Add  more complex person/location config
	std::ifstream input_file(input_file_name);
	json input_json = json::parse(input_file);
	
	int pop_size = input_json.value("population_size", 0);
	int num_locs = input_json.value("num_locations", 0);
	
	std::vector<Location> places;

	srand(time(NULL));

	for(int i = 0; i < num_locs; i++) {
		Location loc;
		loc.interaction_level = 1.;
		places.push_back(loc);
		//TODO: do something with duration function (inheritance?)
	}

	for(int i = 0; i < pop_size; i++) {
		Person person;
		person.infection_status = SUSCEPTIBLE;
		places[rand() % places.size()].people_next_step.push_back(person);
	}

	for(int i = 0; i < num_locs; i++) {
		std::clog << "Location " << i << " has " << places[i].people_next_step.size() << " people." << std::endl;
	}
	
	// Configure disease based on input argument
	json disease_json = input_json.value("disease", input_json);
	Disease disease(disease_json);

	int num_infected = input_json.value("initial_infected", 0);
	int person_to_infect;
	int location_to_infect;

	for(int i = 0; i < num_infected; i++) {
		do {
			location_to_infect = rand() % num_locs;
			person_to_infect = rand() % places[location_to_infect].people_next_step.size();
		} while(places[location_to_infect].people_next_step[person_to_infect].infection_status != SUSCEPTIBLE);
		places[location_to_infect].people_next_step[person_to_infect].infection_status = CARRIER;
		std::clog << location_to_infect << " has an infected person" << std::endl;
	}

	// Susciptible/Infected/Recovered/Deceased
	int num_susceptible = pop_size - num_infected;
	int num_recovered = 0;
	int num_deceased = 0;

	std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	while (num_infected > 0){
		num_infected = num_susceptible = num_recovered = num_deceased = 0;
		for (int loc_idx = 0; loc_idx < places.size(); loc_idx++){
			places[loc_idx].people.swap(places[loc_idx].people_next_step);
			places[loc_idx].people_next_step.clear();
		}

		for (int loc_idx = 0; loc_idx < places.size(); loc_idx++){
			//determine spread of infection from infected to healthy
			std::vector<Person> susceptible_people;
			int num_sick = 0;
			
			// Get number of sick people and set of susceptible people
			for (int person_idx = 0; person_idx < places[loc_idx].people.size(); person_idx++){
				if ((places[loc_idx].people[person_idx].infection_status == SICK) || (places[loc_idx].people[person_idx].infection_status == CARRIER))
					num_sick++;
				//else if (p.infection_status == SUSCEPTIBLE)
				//	susceptible_people.push_back(p);
			}
			
			// Propogate infections in places with infected people
			if(num_sick > 0) {
				for (int person_idx = 0; person_idx < places[loc_idx].people.size(); person_idx++){
					if (places[loc_idx].people[person_idx].infection_status == SUSCEPTIBLE) {
						// TODO: scale infection probability properly
						float infection_probability = disease.SPREAD_FACTOR * places[loc_idx].interaction_level;
						float r = (float) rand() / RAND_MAX;
						if ( r < infection_probability){
							places[loc_idx].people[person_idx].infection_status = CARRIER;
						}
					}
				}
			}

			// Location Movement
			for (int person_idx = 0; person_idx < places[loc_idx].people.size(); person_idx++){
				float r = (float) rand() / RAND_MAX;
				if ( r < MOVEMENT_PROBABILITY){
					int new_loc = rand() % num_locs;
					places[new_loc].people_next_step.push_back( places[loc_idx].people[person_idx] );
				} else {
					places[loc_idx].people_next_step.push_back( places[loc_idx].people[person_idx] );
				}
			}
		}

		// Track Stats and progress infection in people
		for (int loc_idx = 0; loc_idx < places.size(); loc_idx++){
			// Get number of sick people and set of susceptible people
			for (int person_idx = 0; person_idx < places[loc_idx].people_next_step.size(); person_idx++){
				switch (places[loc_idx].people_next_step[person_idx].infection_status){
					case SUSCEPTIBLE:
						num_susceptible++;
						places[loc_idx].people_next_step[person_idx].state_count = 0;
						break;
					case CARRIER:
						num_infected++;

						// TODO: Normal Distribution around average times
						if (places[loc_idx].people_next_step[person_idx].state_count > (int) disease.AVERAGE_INCUBATION_DURATION){
							places[loc_idx].people_next_step[person_idx].infection_status = SICK;
							places[loc_idx].people_next_step[person_idx].state_count = 0;

							// TODO: death rate based on age
							float r = (float) rand() / RAND_MAX;
							if (r < disease.DEATH_RATE)
								places[loc_idx].people_next_step[person_idx].to_die = true;
							else
								places[loc_idx].people_next_step[person_idx].to_die = false;
						} else {
							places[loc_idx].people_next_step[person_idx].state_count++;
						}
						break;
					case SICK:
						num_infected++;

						if (places[loc_idx].people_next_step[person_idx].to_die){
							if (places[loc_idx].people_next_step[person_idx].state_count > disease.AVERAGE_TIME_DEATH)
								places[loc_idx].people_next_step[person_idx].infection_status = DECEASED;
						} else {
							if (places[loc_idx].people_next_step[person_idx].state_count > disease.AVERAGE_TIME_RECOVERY)
								places[loc_idx].people_next_step[person_idx].infection_status = RECOVERED;
						}
						places[loc_idx].people_next_step[person_idx].state_count++;
						break;
					case RECOVERED:
						num_recovered++;
						break;
					case DECEASED:
						num_deceased++;
						break;
					default:
						break;
				}
			}
		}

		std::cout << num_susceptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;
	}
}
