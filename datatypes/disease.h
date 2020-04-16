#ifndef Disease_h
#define Disease_h
#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;

class Disease {
	public:
	float SPREAD_FACTOR;
	float CARRIER_PROBABILITY;
	float AVERAGE_INCUBATION_DURATION;
	float AVERAGE_TIME_DEATH;
	float AVERAGE_TIME_RECOVERY;
	float DEATH_RATE;
	
	Disease(json disease_json);
};

#endif
