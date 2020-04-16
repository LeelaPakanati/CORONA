#include "disease.h"

Disease::Disease(json disease_json){
	float SF = disease_json.value("SPREAD_FACTOR", 0.0);
	float CP = disease_json.value("CARRIER_PROBABILITY", 0.0);
	float AID = disease_json.value("AVERAGE_INCUBATION_DURATION", 0.0);
	float ATD = disease_json.value("AVERAGE_TIME_DEATH", 0.0);
	float ATR = disease_json.value("AVERAGE_TIME_RECOVERY", 0.0);
	float DR = disease_json.value("DEATH_RATE", 0.0);
	

	SPREAD_FACTOR = SF;
	CARRIER_PROBABILITY = CP; 
	AVERAGE_INCUBATION_DURATION = AID;
	AVERAGE_TIME_DEATH = ATD;
	AVERAGE_TIME_RECOVERY = ATR;
	DEATH_RATE = DR;
}
