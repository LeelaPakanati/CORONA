#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <nlohmann/json.hpp>

#include "./datatypes/location.cpp"
#include "./datatypes/person.cpp"
#include "./datatypes/disease.cpp"

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

	srand (time(NULL));

	// TODO: Add  more complex person/location config
	std::ifstream input_file(input_file_name);
	json input_json = json::parse(input_file);
	
	int pop_size = input_json.value("population_size", 0);
	int num_locs = input_json.value("num_locations", 0);
	
	std::vector<Person> people(pop_size);
	std::vector<Location> places(num_locs);

	srand(time(NULL));

	Location loc;
	for(int i = 0; i < num_locs; i++) {
		loc.interaction_level = 1.;
		//TODO: do something with duration function (inheritance?)
	}

	Person person;
	for(int i = 0; i < pop_size; i++) {
		person.infectionStatus = HEALTHY;
		places[rand() % places.size()].people.push_back(person);
		people.push_back(person);
	}
	
	// Configure disease based on input argument
	json disease_json = input_json.value("disease", input_json);
	Disease disease(disease_json);

	int num_infected = input_json.value("initial_infected", 0);
	int person_to_infect;
	for(int i = 0; i < num_infected; i++) {
		do {
			person_to_infect = rand() % pop_size;
		} while(people[person_to_infect].infectionStatus != HEALTHY);
		people[person_to_infect].infectionStatus = CARRIER;
	}

	for(int i = 0; i < num_locs; i++) {
		std::cout << "Location " << i << " has " << places[i].people.size() << " people." << std::endl;
	}

	// Susciptible/Infected/Recovered/Deceased
	int num_susciptible = pop_size - num_infected;
	int num_recovered = 0;
	int num_deceased = 0;

	std::cout << "Susceptible,Infected,Recovered,Deceased" << std::endl;
	while (num_infected > 0){
		for (Location l : places){
			//determine spread of infection from infected to healthy
			std::vector<Person> susceptible_people;
			int num_sick = 0;
			
			// Get number of sick people and set of susceptible people
			for (Person p : people){
				if ((p.infection_status == SICK) || (p.infection_status == CARRIER))
					num_sick++;
				else if (p.infection_status == SUSCEPTIBLE)
					susceptible_people.push_back(p);
			}

			// Determine spread from infected to susceptible
			for (Person p : susceptible_people){
				//TODO: better probability function; incorporate person's affinity to interaction
				float infection_probability = disease.SPREAD_FACTOR * l.interaction_level;
				if (rand() < infection_probability){
					person.infection_status = CARRIER;
				}
			}

			// TODO: determine next location
			for (Person p : people){
				
			}
		}

		for (Person p : people){
			num_infected = num_susceptible = num_recovered = num_deceased = 0;

		 	// generate SIRD stats and progress disease
			// TODO: progression based on bell curve around average time
			switch (p.infection_status){
				case SUSCEPTIBLE:
					num_susceptible++;
					p.state_count = 0;
					break;
				case CARRIER:
					num_infected++;

					if (p.state_count > (int) disease.AVERAGE_INCUBATION_DURATION){
						p.infection_status = SICK;
						p.state_count = 0;
						// TODO: death rate based on age
						if (rand() < disease.DEATH_RATE)
							p.to_die = true;
						else
							p.to_die = false;
					}
					p.state_count++;
					break;
				case SICK:
					num_infected++;

					if (p.to_die){
						if (p.state_count > disease.AVERAGE_TIME_DEATH)
							p.infection_status = DECEASED;
					} else {
						if (p.state_count > disease.AVERAGE_TIME_RECOVERY)
							p.infection_status = RECOVERED;
					}
					p.state_count++;
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

		//std::cout << "Susceptible: " << num_susceptible;
		//std::cout << "\tInfected: " << num_infected;
		//std::cout << "\tRecovered: " << num_recovered;
		//std::cout << "\tDeceased: " << num_deceased;
		//std::cout << std::endl;
		std::cout << num_susceptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;

	}
}
