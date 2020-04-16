#ifndef Location_H
#define Location_H

#include <vector>
#include "person.h"

class Location {
private:
public:
	std::vector<Person*> people;
	std::vector<Person*> people_next_step;
	float interaction_level;
	int duration(Person p);
};

#endif
