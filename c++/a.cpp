#include <iostream>
using namespace std;


class MyClass {
public:
    static int class_var;  // 类作用域变量
};

int MyClass::class_var = 30;

int main()
{
    cout << "Hello, world!" << endl;
    cout << "Hello\tWorld\n\n";
    string greeting = "hello, runoob";
    cout << greeting;
    std::cout << "类变量: " << MyClass::class_var << std::endl;
    return 0;
}