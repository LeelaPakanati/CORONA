#include <iostream>
#include <vector>

#include "./datatypes/location.h"
#include "./datatypes/person.h"
#include "./datatypes/disease.h"

int main(int argc, char** argv){
	//TODO: parse args for input file name 

	std::vector<Person> people;
	std::vector<Location> places;

	//TODO: Parse input file (configuration of people, places, and disease)
	
	//TODO: Generate People and Places based on input
	
	//TODO: Configure disease based on input argument
	Disease disease(.1, .1, 2.5, 6.2, .02);

	// Susciptible/Infected/Recovered/Deceased
	int num_infected = 3; //TODO: get initial infected from input file
	int num_susciptible = people.size() - num_infected;
	int num_recovered = 0;
	int num_deceased = 0;

	while(num_infected>0){
		for(Person p : people){
			//TODO: generate SIRD stats
			//TODO: determine next location
			//TODO: determine disease progression
		}
		std::cout << "Susciptible: " << num_susciptible;
		std::cout << "Infected: " << num_infected;
		std::cout << "Recovered: " << num_recovered;
		std::cout << "Deceased: " << num_deceased;
		std::cout << std::endl;

		for(Location l : places){
			//determine spread of infection from infected to healthy
			std::vector<Person> healthy_people;
			
			for(Person p : people){
				//TODO: get number of infected (perhaps distinguish carrier vs infected) and set of healthy
			}

			for(Person p : healthy_people){
				//TODO: determine probability of infection and determine chance
			}
		}
	}
}
