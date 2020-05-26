//----------------------------------
// File: person.h
// Author: Leela Pakanati, Taylor Chatfield
// Class: ECE497 Cuda Programming
// Assignment: Final Project
// Purpose: Person Struct
// Date: 5/25/2020
//----------------------------------

#ifndef PERSON_H
#define PERSON_H

#include "infectionStatus.cpp"
#include "location.h"

struct Location;

struct Person {
	uint id;
	INFECTION_STATUS infection_status;
	uint state_count;
	bool to_die;
};

#endif //PERSON_H
