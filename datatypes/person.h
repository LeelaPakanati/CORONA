#ifndef PERSON_H
#define PERSON_H

#include "infectionStatus.cpp"
#include "location.h"

struct Location;

struct Person {
	int id;
	INFECTION_STATUS infection_status;
	int state_count;
	bool to_die;
};

#endif //PERSON_H
