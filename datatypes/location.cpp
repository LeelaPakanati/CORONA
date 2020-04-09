#include "person.cpp"

class Location {
private:
public:
	std::vector<Person> people;
	float interaction;
	virtual int duration(Person p);
