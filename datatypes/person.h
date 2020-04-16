#ifndef PERSON_H
#define PERSON_H

#include "infectionStatus.cpp"
#include "location.h"

class Location;

class Person {
private:
public:
	Location * home;
	Location * work;
	INFECTION_STATUS infection_status;
	int state_count;
	bool to_die;
};

#endif //PERSON_H
