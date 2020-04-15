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
	if (argc < 2){
		std::cerr << "Usage : " << argv[0] << " <input file>" << std::endl;
		return 0;
	}

	std::string input_file_name = argv[1];
	std::clog << "Reading file " << input_file_name << " for starting conditions" << std::endl;

	//TODO: Add disease configuration and more complex person/location config
	std::ifstream input_file(input_file_name);
	json input_json = json::parse(input_file);
	
	//TODO: Generate People and Places based on input
	int pop_size = input_json.value("population_size", 0);
	int num_locs = input_json.value("num_locations", 0);
	
	std::vector<Person> people(pop_size);
	std::vector<Location> places(num_locs);
	
	//TODO: Configure disease based on input argument
	json disease_json = input_json.value("disease", input_json);
	float SF = disease_json.value("SPREAD_FACTOR", 0.0);
	float CP = disease_json.value("CARRIER_PROBABILITY", 0.0);
	float AID = disease_json.value("AVERAGE_INCUBATION_DURATION", 0.0);
	float ATD = disease_json.value("AVERAGE_TIME_DEATH", 0.0);
	float ATR = disease_json.value("AVERAGE_TIME_RECOVERY", 0.0);
	float DR = disease_json.value("DEATH_RATE", 0.0);
	
	Disease disease(SF, CP, AID, ATD, ATR, DR);

	// Susciptible/Infected/Recovered/Deceased
	int num_infected = input_json.value("initial_infected", 0); //TODO: get initial infected from input file
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
