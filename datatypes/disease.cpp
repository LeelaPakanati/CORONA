#include "disease.h"

Disease::Disease(float spread_factor, float carrier_probability, float average_incubation_duration, float average_infection_duration, float death_probability){
	SPREAD_FACTOR = spread_factor;
	CARRIER_PROBABILITY = carrier_probability; 
	AVERAGE_INFECTION_DURATION = average_infection_duration;
	AVERAGE_INCUBATION_DURATION = average_incubation_duration;
	DEATH_PROBABILITY = death_probability;
}
