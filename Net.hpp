//
//  Net.hpp
//  test1118
//
//  Created by 许清嘉 on 11/19/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef Net_hpp
#define Net_hpp

#include <iostream>
#include <vector>
#include <list>
#include "Layer.hpp"
#define DEFAULT_VALUE 999
using namespace std;

class Net{
public:
    vector<Layer> layerVector;
    string activateFunction;
    string lossFunction;
    float delta;//for adjusting Huber loss Func
    float learningRate;
    vector<float> output;//expected output
    int maxIteration;
    int currentIteration;
    vector<float> direction;//record the sign of the difference between initial output and expected output
    vector<int> learningRateUpdate;//record the iteration where learning rate halves
    
    Net(list<int>, list<float>, list<float>);
    
    void initializeNet(float fixed = DEFAULT_VALUE, float bias = 0);
    void initializeNetFirstVersion(float fixed = DEFAULT_VALUE, float bias = 0);
    void testInitialize();
    void forwardPropagation(int);
    void lossComputation();
//    void testFindOptimalLearningRate(int);
    void backwardPropagation(int);
    void train(float learning = 0.5, int max = 5000, string activate = "sigmoid", string loss = "squared", float d = 0.5);
    
    
};

#endif /* Net_hpp */
