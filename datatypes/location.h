#ifndef Location_H
#define Location_H

#include <vector>
#include "person.h"

class Location {
private:
public:
	std::vector<Person> people;
	float interaction_level;
	int duration(Person p);
};

#endif
