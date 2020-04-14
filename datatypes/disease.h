#ifndef Disease_h
#define Disease_h

class Disease {
	public:
	float SPREAD_FACTOR;
	float CARRIER_PROBABILITY;
	float AVERAGE_INCUBATION_DURATION;
	float AVERAGE_TIME_DEATH;
	float AVERAGE_TIME_RECOVERY;
	float DEATH_RATE;
	
	Disease(float spread_factor, float carrier_probability, float average_incubation_duration, float average_time_death, float average_time_recovery, float death_rate);
};

#endif
