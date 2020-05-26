//----------------------------------
// File: diseas.h
// Author: Leela Pakanati, Taylor Chatfield
// Class: ECE497 Cuda Programming
// Assignment: Final Project
// Purpose: Disease Struct
// Date: 5/25/2020
//----------------------------------

#ifndef DISEASE_H
#define DISEASE_H

struct Disease {
	float SPREAD_FACTOR;
	float CARRIER_PROBABILITY;
	float AVERAGE_INCUBATION_DURATION;
	float AVERAGE_TIME_DEATH;
	float AVERAGE_TIME_RECOVERY;
	float DEATH_RATE;
};

#endif //DISEASE_H
