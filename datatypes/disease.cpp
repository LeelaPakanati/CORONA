class Disease {
	public:
	float SPREAD_FACTOR;
	float CARRIER_PROBABILITY;
	float AVERAGE_INCUBATION_DURATION;
	float AVERAGE_INFECTION_DURATION;
	float DEATH_PROBABILITY;
	
	Disease(float spread_factor, carrier_probability, average_incubation_duration, average_infection_duration, death_probability){
		SPREAD_FACTOR = spread_factor;
		CARRIER_PROBABILITY = carrier_probability; 
		AVERAGE_INFECTION_DURATION = average_infection_duration;
		AVERAGE_INCUBATION_DURATION = average_incubation_duration;
		DEATH_PROBABILITY = death_probability;
	}
}
