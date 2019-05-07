#include <iostream>

#include <adamcuda.h>
#include <iostream>

class Container {
    int * m_Data;
public:
    Container() {
        //Allocate an array of 20 int on heap
        m_Data = new int[20];

        std::cout << "Constructor: Allocation 20 int" << std::endl;
    }
    ~Container() {
        if (m_Data) {
            delete[] m_Data;
            m_Data = NULL;
        }
    }

    // Copy constructor
    Container(const Container & obj) {
        //Allocate an array of 20 int on heap
        m_Data = new int[20];

        //Copy the data from passed object
        for (int i = 0; i < 20; i++)
            m_Data[i] = obj.m_Data[i];

        std::cout << "Copy Constructor: Allocation 20 int" << std::endl;
    }

    //Assignment Operator
    Container & operator=(const Container & obj) {

        if(this != &obj)
        {
            //Allocate an array of 20 int on heap
            m_Data = new int[20];

            //Copy the data from passed object
            for (int i = 0; i < 20; i++)
                m_Data[i] = obj.m_Data[i];

            std::cout << "Assigment Operator: Allocation 20 int" << std::endl;
        }
    }

    // Move Constructor
    Container(Container && obj){
        // Just copy the pointer
        m_Data = obj.m_Data;

        // Set the passed object's member to NULL
        obj.m_Data = NULL;

        std::cout<<"Move Constructor"<<std::endl;
    }

    // Move Assignment Operator
    Container& operator=(Container && obj){
        if(this != &obj)
        {
            // Just copy the pointer
            m_Data = obj.m_Data;

            // Set the passed object's member to NULL
            obj.m_Data = NULL;

            std::cout<<"Move Assignment Operator"<<std::endl;
        }
    }
};

Container getContainer(){
    Container c;
    return c;
}


namespace raaj{
    template<typename T>
    class shared_ptr{
    public:
        T* object_;
        unsigned int* reference_counter_ = nullptr;
        std::mutex lock_;

        unsigned int use_count(){
            return *reference_counter_;
        }

        T* get(){
            return object_;
        }

        shared_ptr(){
            std::cout << "Empty Constructor" << std::endl;
            reference_counter_ = new unsigned int[1];
            *reference_counter_ = 0;
        }

        // Normal Constructor
        shared_ptr(T* object){
            std::cout << "Normal Constructor" << std::endl;
            object_ = object;
            reference_counter_ = new unsigned int[1];
            *reference_counter_ = 1;
        }

        // Copy Constructor (CANT USE CONST HERE)
        shared_ptr(const shared_ptr & obj) {
            std::cout << "Copy Constructor" << std::endl;

            //lock_.lock();
            // increase ref
            *obj.reference_counter_ += 1;
            // set reset
            reference_counter_ = obj.reference_counter_;
            object_ = obj.object_;
            //lock_.unlock();
        }

        //Assignment Operator
        shared_ptr & operator=(const shared_ptr & obj) {
            if(this != &obj)
            {
                std::cout << "Assignment Operator" << std::endl;

                // increase ref
                *obj.reference_counter_ += 1;
                // set reset
                //delete reference_counter_;
                reference_counter_ = obj.reference_counter_; // I am overriding this?
                object_ = obj.object_;
            }
        }


        // Destructor
        ~shared_ptr(){
            std::cout << "Destructor" << std::endl;
            *reference_counter_ -= 1;
            if(use_count() == 0){
                delete object_;
                delete reference_counter_;
            }
        }

    };
}

class Object{
  public:
    Object(){}
    ~Object(){}
};

////        // Move Constructor
////        shared_ptr(shared_ptr && obj){
////            std::cout<<"Move Constructor"<<std::endl;
////        }

//        // Move Assignment Operator
//        shared_ptr& operator=(shared_ptr && obj){
//            if(this != &obj)
//            {
//                std::cout<<"Move Assignment Operator"<<std::endl;
//            }
//        }


void function(raaj::shared_ptr<Object> object_ptr)
{
    std::cout << object_ptr.use_count() << std::endl;
    //object_ptr.reset();
}


int main(int argc, char* argv[])
{

//    // Create a vector of Container Type
//    std::cout << "-1-" << std::endl;
//    std::vector<Container> vecOfContainers;
//    std::cout << "-2-" << std::endl;
//    Container cont; // Regular Constructor
//    std::cout << "-3-" << std::endl;
//    vecOfContainers.push_back(cont); // Copy Constructor - Copy data from passed object
//    std::cout << "-4-" << std::endl;
//    vecOfContainers.clear(); //
//    std::cout << "-5-" << std::endl;
//    vecOfContainers.reserve(3);
//    std::cout << "-6-" << std::endl;
//    vecOfContainers.push_back(Container()); // Constructor, then Move (just set the pointer directly)
//    std::cout << "-7-" << std::endl;
//    //Container cont2 = cont;
//    cont = getContainer(); // Constructor, then Move Assignment (For return operation, same as above)
//    std::cout << "-8-" << std::endl;
//    Container contx;
//    contx = cont; // Assignment Operator // Copy data from passed object if not the same

//    exit(-1);

    ////////////////////



    raaj::shared_ptr<Object> object_ptr = raaj::shared_ptr<Object>(new Object());

    // Copy Constructor
    {
        raaj::shared_ptr<Object> object_ptr_ref = object_ptr;
        std::cout << object_ptr.use_count() << std::endl;
        function(object_ptr);
    }

    // Assignment Operator
    {
        raaj::shared_ptr<Object> hold;
        hold = object_ptr;
        std::cout << object_ptr.use_count() << std::endl;
    }

    std::cout << object_ptr.use_count() << std::endl;


    ////////////////////

//    std::shared_ptr<Object> object_ptr = std::shared_ptr<Object>(new Object());

//    {
//        std::shared_ptr<Object> object_ptr_ref = object_ptr;
//        std::cout << object_ptr.use_count() << std::endl;
//    }

//    function(object_ptr);

//    std::cout << object_ptr.use_count() << std::endl;


    return 0;
}

