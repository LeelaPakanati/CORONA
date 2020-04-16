#ifndef Person_H
#define Person_H

#include "infectionStatus.cpp"

class Person {
private:
public:
	INFECTION_STATUS infection_status;
	int state_count;
	bool to_die;
};



#endif
