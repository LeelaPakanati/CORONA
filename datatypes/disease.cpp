#include "disease.h"

Disease::Disease(float spread_factor, float carrier_probability, float average_incubation_duration, float average_time_death, float average_time_recovery, float death_rate){
	SPREAD_FACTOR = spread_factor;
	CARRIER_PROBABILITY = carrier_probability; 
	AVERAGE_INCUBATION_DURATION = average_incubation_duration;
	AVERAGE_TIME_DEATH = average_time_death;
	AVERAGE_TIME_RECOVERY = average_time_recovery;
	DEATH_RATE = death_rate;
}
