#ifndef LOCATION_H
#define LOCATION_H

#include <vector>
#include "person.h"

class Person;

class Location {
private:
public:
	std::vector<Person *> people;
	std::vector<Person *> people_next_step;
	float interaction_level;
	int duration(Person p);
};

#endif //LOCATION_H
