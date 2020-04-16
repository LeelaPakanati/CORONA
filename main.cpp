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
	if (argc < 2){
		std::cerr << "Usage : " << argv[0] << " <input file>" << std::endl;
		return 0;
	}

	std::string input_file_name = argv[1];
	std::clog << "Reading file " << input_file_name << " for starting conditions" << std::endl;

	//TODO: Add disease configuration and more complex person/location config
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
	
	//TODO: Configure disease based on input argument
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
	while (num_infected>0){
		for (Person p : people){
			//TODO: generate SIRD stats
			//TODO: determine next location
			//TODO: determine disease progression
		}
		// test pattern
		if((num_infected < num_susciptible) && (num_deceased == 0)){
			num_infected++;
			num_susciptible--;
		} else{
			num_deceased++;
			num_infected--;
		}
		//std::cout << "Susciptible: " << num_susciptible;
		//std::cout << "\tInfected: " << num_infected;
		//std::cout << "\tRecovered: " << num_recovered;
		//std::cout << "\tDeceased: " << num_deceased;
		//std::cout << std::endl;
		std::cout << num_susciptible << "," << num_infected << "," << num_recovered << "," << num_deceased << std::endl;

		for (Location l : places){
			//determine spread of infection from infected to healthy
			std::vector<Person> healthy_people;
			
			for (Person p : people){
				//TODO: get number of infected (perhaps distinguish carrier vs infected) and set of healthy
			}

			for (Person p : healthy_people){
				//TODO: determine probability of infection and determine chance
			}
		}
	}
}
