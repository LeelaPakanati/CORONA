#ifndef Disease_h
#define Disease_h

class Disease {
	public:
	float SPREAD_FACTOR;
	float CARRIER_PROBABILITY;
	float AVERAGE_INCUBATION_DURATION;
	float AVERAGE_INFECTION_DURATION;
	float DEATH_PROBABILITY;
	
	Disease(float spread_factor, float carrier_probability, float average_incubation_duration, float average_infection_duration, float death_probability);
};

#endif
