#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <nlohmann/json.hpp>

#include "./datatypes/location.cpp"
#include "./datatypes/person.cpp"
#include "./datatypes/disease.cpp"

// temporary
#define MOVEMENT_PROBABILITY .1

// for convenience
using json = nlohmann::json;

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
	
	// All other references to these objects should be pointers or arrays of pointers
	std::vector<Location> places;
	std::vector<Person> people;

	srand(time(NULL));

	Location *loc_ptr;
	for(int i = 0; i < num_locs; i++) {
		Location loc;
		loc_ptr = &loc;
		loc_ptr->interaction_level = 1.;
		places.push_back(*loc_ptr);
		//TODO: do something with duration function (inheritance?)
	}

	Person *person_ptr;
	for(int i = 0; i < pop_size; i++) {
		Person person;
		person_ptr = &person;
		person_ptr->infection_status = SUSCEPTIBLE;
		people.push_back(*person_ptr);
		places[rand() % places.size()].people_next_step.push_back(person_ptr);
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

	// Infect initial population.
	// note: people in less populated locations are more likely to be infected.
	// This should only matter for initially infecting large (>25%?) amounts of the population.
	for(int i = 0; i < num_infected; i++) {
		do {
			location_to_infect = rand() % num_locs;
			person_to_infect = rand() % places[location_to_infect].people_next_step.size();
		} while(people[person_to_infect].infection_status != SUSCEPTIBLE);
		people[person_to_infect].infection_status = CARRIER;
		std::clog << location_to_infect << " has an infected person" << std::endl;
	}

	// Susciptible/Infected/Recovered/Deceased
	int num_susceptible = pop_size - num_infected;
	int num_recovered = 0;
	int num_deceased = 0;

	std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	while (num_infected > 0) {
		num_infected = num_susceptible = num_recovered = num_deceased = 0;

		// Update location changes
		for (int loc_idx = 0; loc_idx < places.size(); loc_idx++) {
			loc_ptr = &places[loc_idx];
			loc_ptr->people.swap(loc_ptr->people_next_step);
			loc_ptr->people_next_step.clear();
		}

		// Spread disease
		for (int loc_idx = 0; loc_idx < places.size(); loc_idx++) {
			loc_ptr = &places[loc_idx];

			//determine spread of infection from infected to healthy
			std::vector<Person*> susceptible_people;
			int num_sick = 0;
			
			// Get number of sick people
			for (int person_idx = 0; person_idx < loc_ptr->people.size(); person_idx++) {
				person_ptr = loc_ptr->people[person_idx];
				if ((person_ptr->infection_status == SICK) || (person_ptr->infection_status == CARRIER)) {
					num_sick++;
				}
			}
			
			// Propogate infections in places with infected people
			if(num_sick > 0) {
				for (int person_idx = 0; person_idx < loc_ptr->people.size(); person_idx++) {
					person_ptr = loc_ptr->people[person_idx];
					if (person_ptr->infection_status == SUSCEPTIBLE) {
						// TODO: scale infection probability properly
						float infection_probability = disease.SPREAD_FACTOR * loc_ptr->interaction_level;
						float r = (float) rand() / RAND_MAX;
						if (r < infection_probability) {
							person_ptr->infection_status = CARRIER;
						}
					}
				}
			}
		}

		// Determine next locations
		for(int person_idx = 0; person_idx < people.size(); person_idx++){
			person_ptr = &people[person_idx];
			float r = (float) rand() / RAND_MAX;
			if(r < MOVEMENT_PROBABILITY) {
				int new_loc = rand() % num_locs;
				places[new_loc].people_next_step.push_back( person_ptr );
			} else {
				loc_ptr->people_next_step.push_back( person_ptr );
			}
		}

		// Collect statistics
		for (int loc_idx = 0; loc_idx < places.size(); loc_idx++) {
			loc_ptr = &places[loc_idx];
			// Get number of sick people and set of susceptible people
			for (int person_idx = 0; person_idx < loc_ptr->people_next_step.size(); person_idx++) {
				person_ptr = loc_ptr->people_next_step[person_idx];
				switch (person_ptr->infection_status) {
					case SUSCEPTIBLE:
						num_susceptible++;
						person_ptr->state_count = 0;
						break;
					case CARRIER:
						num_infected++;

						// TODO: Normal Distribution around average times
						if (person_ptr->state_count > (int) disease.AVERAGE_INCUBATION_DURATION) {
							person_ptr->infection_status = SICK;
							person_ptr->state_count = 0;

							// TODO: death rate based on age
							float r = (float) rand() / RAND_MAX;
							if (r < disease.DEATH_RATE)
								person_ptr->to_die = true;
							else
								person_ptr->to_die = false;
						} else {
							person_ptr->state_count++;
						}
						break;
					case SICK:
						num_infected++;

						if (person_ptr->to_die) {
							if (person_ptr->state_count > disease.AVERAGE_TIME_DEATH)
								person_ptr->infection_status = DECEASED;
						} else {
							if (person_ptr->state_count > disease.AVERAGE_TIME_RECOVERY)
								person_ptr->infection_status = RECOVERED;
						}
						person_ptr->state_count++;
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
