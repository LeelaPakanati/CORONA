#include "person.cpp"

class Location {
private:
public:
	std::vector<Person> people;
	float interaction_level;
	virtual int duration(Person p);
}
