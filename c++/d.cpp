#include <iostream>

struct MyClass {
    int a;
    MyClass() { std::cout << "Constructor called\n"; a = 42; }
    MyClass(const MyClass& other) { std::cout << "Copy constructor called\n"; a = other.a; }
    MyClass& operator=(const MyClass& other) {
        std::cout << "Assignment operator called\n";
        a = other.a;
        return *this;
    }
};

int main() {
    void* mem = malloc(sizeof(MyClass));

    // placement new
    MyClass* p1 = new (mem) MyClass();  
    std::cout << "1---------------------\n";
    
    MyClass* p2 = new MyClass();
    std::cout << "2---------------------\n";
    
    MyClass p3 = MyClass();
    // 强转 + 赋值
    std::cout << "3---------------------\n";
    
    MyClass* p4 = new (mem) MyClass(*p2);
    std::cout << "4---------------------\n";
    
    MyClass* p5 = (MyClass*)mem;
    *p2 = MyClass();  

    p1->~MyClass();
    free(mem);
}
