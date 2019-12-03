//
//  main.cpp
//  test1118
//
//  Created by 许清嘉 on 11/18/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

//BP theory illustration at: http://baijiahao.baidu.com/s?id=1603526007545961785&wfr=spider&for=pc
#include <ctime>
#include "Layer.hpp"
#include "Net.hpp"

using namespace std;

void testSimplify(){
    //param1: node num of each layer, param2: input of first layer, param3: expected output of last layer
    Net net({2, 2, 2}, {0.1, 0.2}, {0.01, 0.99});
    //the expected output should be greater than -1
    //the expected output should be less than 1 as well
    net.testInitialize();
    //param1: learningRate, param2: maxIteration, param3: activateFuction, param4: lossFunction
    net.train(23, 1750);
}

void testRandom(){
    
//    Net net({3, 2, 3, 2}, {0.1, 0.2, 0.3}, {0.01, 0.99});
    Net net({2, 3, 2}, {0.1, 0.2}, {0.4, 0.6});
    net.initializeNet();//param1: node Weight, param2: bias Weight
    net.train(3, 50, "sigmoid", "L2");
    //L1 is a loss function more sensitive to learning rate, so the learning rate will be reduced by half plenty times
    //theoretically, the optimal learning rate should be the d(loss_function) / d(learning_rate), but the relationship between loss_function and learning_rate is a complex composite function that is very hard to solve.
}

int main(int argc, const char * argv[]) {
    // insert code here...
    clock_t startTime, endTime;
    startTime = clock();
    testRandom();
    endTime = clock();
    cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}


