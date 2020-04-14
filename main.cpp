#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>


#include "./datatypes/location.cpp"
#include "./datatypes/person.cpp"
#include "./datatypes/disease.cpp"

// for convenience
using json = nlohmann::json;

int main(int argc, char** argv){
	//TODO: parse args for input file name 
	if (argc < 2){
		std::cerr << "Usage : " << argv[0] << " <input file>" << std::endl;
		return 0;
	}

	std::string input_file_name = argv[1];
	std::cout << "Reading file " << input_file_name << " for starting conditions" << std::endl;

	//TODO: Parse input file (configuration of people, places, and disease)
	std::ifstream input_file(input_file_name);
	json jf = json::parse(input_file);
	
	int pop_size = jf.value("population_size", 0);
	int num_locs = jf.value("num_locations", 0);

	std::cout << "Num Population " << pop_size << "\tNum Locations " << num_locs << std::endl;
	
	//TODO: Generate People and Places based on input
	std::vector<Person> people(pop_size);
	std::vector<Location> places(num_locs);
	
	//TODO: Configure disease based on input argument
	Disease disease(.1, .1, 2.5, 6.2, .02);

	// Susciptible/Infected/Recovered/Deceased
	int num_infected = jf.value("initial_infected", 0); //TODO: get initial infected from input file
	int num_susciptible = pop_size - num_infected;
	int num_recovered = 0;
	int num_deceased = 0;

	while (num_infected>0){
		for (Person p : people){
			//TODO: generate SIRD stats
			//TODO: determine next location
			//TODO: determine disease progression
		}
		std::cout << "Susciptible: " << num_susciptible;
		std::cout << "\tInfected: " << num_infected;
		std::cout << "\tRecovered: " << num_recovered;
		std::cout << "\tDeceased: " << num_deceased;
		std::cout << std::endl;

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
