#include <iostream>

struct bar{
	int b;
};

struct foo{
	bar arr[5000];
	int a;
};


int main(){
	std::cout << sizeof(struct foo) << std::endl;
	struct foo* a = (foo *) malloc(10 * sizeof(struct foo));
	foo* foo_ptr = &a[0];
	foo_ptr->a = 5;
	std::cout << a[0].a << std::endl;
	bar* bar_ptr = &foo_ptr->arr[1000];
	bar_ptr->b = 100;
	std::cout << a[0].arr[1000].b << std::endl;
}
