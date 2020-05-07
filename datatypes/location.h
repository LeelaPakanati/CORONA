#ifndef LOCATION_H
#define LOCATION_H

#include "person.h"

struct Person;

#define MAX_LOCATION_CAPACITY 256

struct Location {
	Person people[MAX_LOCATION_CAPACITY];
	int num_people;
	Person people_next_step[MAX_LOCATION_CAPACITY];
	int num_people_next_step;
	float interaction_level;
};

#endif //LOCATION_H
