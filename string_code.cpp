enum string_code {
	e_debug,
	e_population_size,
	e_num_locations,
	e_max_size,
	e_initial_infected,
	e_SPREAD_FACTOR,
	e_CARRIER_PROBABILITY,
	e_AVERAGE_INCUBATION_DURATION,
	e_AVERAGE_TIME_DEATH,
	e_AVERAGE_TIME_RECOVERY,
	e_DEATH_RATE,
	e_null
};

string_code hash_it (std::string const& inString) {
	if (inString == "debug") return e_debug;
	else if (inString == "population_size") return e_population_size;
	else if (inString == "num_locations") return e_num_locations;
	else if (inString == "max_size") return e_max_size;
	else if (inString == "initial_infected") return e_initial_infected;
	else if (inString == "SPREAD_FACTOR") return e_SPREAD_FACTOR;
	else if (inString == "CARRIER_PROBABILITY") return e_CARRIER_PROBABILITY;
	else if (inString == "AVERAGE_INCUBATION_DURATION") return e_AVERAGE_INCUBATION_DURATION;
	else if (inString == "AVERAGE_TIME_DEATH") return e_AVERAGE_TIME_DEATH;
	else if (inString == "AVERAGE_TIME_RECOVERY") return e_AVERAGE_TIME_RECOVERY;
	else if (inString == "DEATH_RATE") return e_DEATH_RATE;
	else return e_null;
}
