//
//  Net.cpp
//  test1118
//
//  Created by 许清嘉 on 11/19/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "Net.hpp"
#include <random>
#include <cmath>
#include <assert.h>
#include <algorithm>

const list<string> activateList = {"sigmoid", "tanh", "ReLU", "LeakyReLU"};
const list<string> lossList = {"squared", "L1", "L2", "Huber", "Log-Cosh"};

Net::Net(list<int> layerNodeNum, list<float> in, list<float> out){
    assert(layerNodeNum.front() == in.size());
    assert(layerNodeNum.back() == out.size());
    currentIteration = 0;
    for (auto iter = layerNodeNum.begin(); iter != layerNodeNum.end(); iter++){
        Layer layer = Layer(*iter);
        layerVector.push_back(layer);
    }
    layerVector.back().setIsOutputLayer(true);
    vector<float> input;
    for (auto iter = in.begin(); iter != in.end(); iter++){
        input.push_back(*iter);
    }
    layerVector.front().setOut(input);
    for (auto iter = out.begin(); iter != out.end(); iter++){
        output.push_back(*iter);
        if (*iter < -1 || *iter > 1)
//            perror("The expected output ");
            cout << "===========[Warning]==========\nThe expected output " << *iter << " should be between 0 and 1 for better result." << endl;
    }
    
}

void Net::initializeNet(float fixed, float bias){
    if (fixed == DEFAULT_VALUE){
        //assign random value
        default_random_engine e;
//        uniform_real_distribution<float> u(0, 1); //(plan first)uniform distribution object
//        normal_distribution<float> u(0.5, 0.5); //normal distribution object, param1: mean, param2: standard deviation
        for (auto iter = layerVector.begin(); iter != layerVector.end() - 1; iter++){
            int row = iter->getNodeNum();
            int column = (iter + 1)->getNodeNum() - 1;
            vector<vector<float>> weight;
            for (int i = 0; i < row; i++){
                vector<float> subWeight;
//                normal_distribution<float> u(0, 1);//plan A
                uniform_real_distribution<float> u(0, 1.0 / column);//plan B
                for (int j = 0; j < column; j++){
//                    subWeight.push_back(u(e) / column);//plan A
                    subWeight.push_back(u(e));
                }
                weight.push_back(subWeight);
            }
            iter->setWeight(weight);
        }
    }
    else{
        //assign user-specified value for all weights
        if (bias == 0)
            bias = fixed;
        for (int k = 0; k < layerVector.size() - 1; k++){
            int row = layerVector.at(k).getNodeNum();
            int column = layerVector.at(k + 1).getNodeNum() - 1;
            vector<vector<float>> weight;
            for (int i = 0; i < row - 1; i++){
                vector<float> subWeight;
                for (int j = 0; j < column; j++){
                    subWeight.push_back(fixed);
                }
                weight.push_back(subWeight);
            }
            vector<float> biasWeight;
            for (int j = 0; j < column; j++)
                biasWeight.push_back(bias);
            weight.push_back(biasWeight);
            layerVector[k].setWeight(weight);
        }
    }
}

void Net::initializeNetFirstVersion(float fixed, float bias){
    if (fixed == DEFAULT_VALUE){
        //assign random value
        default_random_engine e;
        uniform_real_distribution<float> u(0, 1); //(plan first)uniform distribution object
//        normal_distribution<float> u(0.5, 0.5); //normal distribution object, param1: mean, param2: standard deviation
        for (auto iter = layerVector.begin(); iter != layerVector.end() - 1; iter++){
            int row = iter->getNodeNum();
            int column = (iter + 1)->getNodeNum() - 1;
            vector<vector<float>> weight;
            for (int i = 0; i < row; i++){
                vector<float> subWeight;
                for (int j = 0; j < column; j++){
                    subWeight.push_back(u(e));
                }
                weight.push_back(subWeight);
            }
            iter->setWeight(weight);
        }
    }
    else{
        //assign user-specified value for all weights
        if (bias == 0)
            bias = fixed;
        for (int k = 0; k < layerVector.size() - 1; k++){
            int row = layerVector.at(k).getNodeNum();
            int column = layerVector.at(k + 1).getNodeNum() - 1;
            vector<vector<float>> weight;
            for (int i = 0; i < row - 1; i++){
                vector<float> subWeight;
                for (int j = 0; j < column; j++){
                    subWeight.push_back(fixed);
                }
                weight.push_back(subWeight);
            }
            vector<float> biasWeight;
            for (int j = 0; j < column; j++)
                biasWeight.push_back(bias);
            weight.push_back(biasWeight);
            layerVector[k].setWeight(weight);
        }
    }
}

void Net::testInitialize(){
    vector<vector<float>> weight1;
    vector<float> weight11;
    weight11.push_back(0.1);
    weight11.push_back(0.3);
    vector<float> weight12;
    weight12.push_back(0.2);
    weight12.push_back(0.4);
    vector<float> weight13;
    weight13.push_back(0.55);
    weight13.push_back(0.56);
    weight1.push_back(weight11);
    weight1.push_back(weight12);
    weight1.push_back(weight13);
    
    layerVector.at(0).setWeight(weight1);
    
    vector<vector<float>> weight2;
    vector<float> weight21;
    weight21.push_back(0.5);
    weight21.push_back(0.7);
    vector<float> weight22;
    weight22.push_back(0.6);
    weight22.push_back(0.8);
    vector<float> weight23;
    weight23.push_back(0.66);
    weight23.push_back(0.67);
    weight2.push_back(weight21);
    weight2.push_back(weight22);
    weight2.push_back(weight23);
    
    layerVector.at(1).setWeight(weight2);
}

void Net::forwardPropagation(int layerNo){
    if (layerNo > layerVector.size() - 2)
        return;
    Layer layer = layerVector.at(layerNo);
    Layer renewLayer = layerVector.at(layerNo + 1);
    vector<vector<float>> layerWeight = layer.getWeight();
    vector<float> layerOut = layer.getOut();
    vector<float> renewLayerOut;
    int row = layer.getNodeNum();
    int column = renewLayer.getNodeNum() - 1;
    for (int j = 0; j < column; j++){
        float in = 0.0, out;
        for (int i = 0; i < row; i++){
            in += layerWeight.at(i).at(j) * layerOut.at(i);
        }
        if (activateFunction == "sigmoid"){
            out = 1 / (exp(-in) + 1);
        }
        else if (activateFunction == "tanh"){
            out = tanh(in);
        }
        else if (activateFunction == "ReLU"){//Rectified Linear Unit
            if (in < 0)
                out = 0;
            else out = in;
        }
        else if (activateFunction == "LeakyReLU"){
            if (in < 0)
                out = 0.01 * in;
            else out = in;
        }
        renewLayerOut.push_back(out);
    }
    layerVector[layerNo + 1].setOut(renewLayerOut);
}

void Net::lossComputation(){
    vector<float> layerOut = layerVector.back().getOut();
    float loss = 0;
    if (direction.size() == 0){
        for (int i = 0; i < output.size(); i++){
            direction.push_back(layerOut.at(i) - output.at(i));
        }
    }
    else {
        bool flag = false;
        for (int i = 0; i < output.size(); i++){
            if ((layerOut.at(i) - output.at(i)) * direction.at(i) < 0){
                cout << "Warning: learning Rate is too big for output " << i << endl;
                direction[i] = layerOut.at(i) - output.at(i);
                flag = true;
            }
        }
        if (flag){
            learningRate /= 2;
            learningRateUpdate.push_back(currentIteration + 1);
            cout << "learning rate: " << learningRate << endl;
        }
    }
    if (lossFunction == "squared"){
        for (int i = 0; i < output.size(); i++){
            loss += pow(layerOut.at(i) - output.at(i), 2);
        }
        loss /= 2;
    }
    else if (lossFunction == "L1"){
        for (int i = 0; i < output.size(); i++){
            loss += abs(layerOut.at(i) - output.at(i));
        }
        loss /= output.size();
    }
    else if (lossFunction == "L2"){
        for (int i = 0; i < output.size(); i++){
            loss += pow(layerOut.at(i) - output.at(i), 2);
        }
        loss /= 2 * output.size();
    }
    else if (lossFunction == "Huber"){
        for (int i = 0; i < output.size(); i++){
            if (abs(output.at(i) - layerOut.at(i)) <= delta){
                loss += pow(layerOut.at(i) - output.at(i), 2) / 2;
            }
            else {
                loss += delta * abs(output.at(i) - layerOut.at(i)) - pow(delta, 2) / 2;
            }
        }
    }
    else if (lossFunction == "Log-Cosh"){
        for (int i = 0; i < output.size(); i++){
            loss += log(cosh(layerOut.at(i) - output.at(i)));
        }
    }
    cout << "loss = " << loss << ", iteration(" << currentIteration + 1 << "/" << maxIteration << ")" << endl;
}

void Net::backwardPropagation(int layerNo){
    if (layerNo < 1)
        return;
    Layer layer = layerVector.at(layerNo);
    Layer renewLayer = layerVector.at(layerNo - 1);
    vector<float> layerOut = layer.getOut();
    vector<float> renewLayerOut = renewLayer.getOut();
    vector<vector<float>> renewWeight = renewLayer.getWeight();
    vector<float> coef;
    int row = renewLayer.getNodeNum() - 1;
    int column = layer.getNodeNum() - 1;
    if (layer.getIsOutputLayer() == true){
        for (int j = 0; j < column; j++){
            float subCoef = 1;
            if (lossFunction == "squared"){
                subCoef *= -(output.at(j) - layerOut.at(j));
            }
            else if (lossFunction == "L2"){
                subCoef *= -(output.at(j) - layerOut.at(j)) / output.size();
            }
            else if (lossFunction == "L1"){
                if (layerOut.at(j) < output.at(j))
                    subCoef = -1;
                else if (layerOut.at(j) > output.at(j))
                    subCoef = 1;
                else
                    subCoef = 0;//set a derivative for the nondifferential point 0
            }
            else if (lossFunction == "Huber"){
                if (abs(output.at(j) - layerOut.at(j)) <= delta){
                    subCoef *= -(output.at(j) - layerOut.at(j));
                }
                else if (layerOut.at(j) - output.at(j) > 0)
                    subCoef *= delta;
                else
                    subCoef *= -delta;
            }
            else if (lossFunction == "Log-Cosh"){
                subCoef *= tanh(layerOut.at(j) - output.at(j)) * layerOut.at(j);
            }
            
            if (activateFunction == "sigmoid"){
                subCoef *= layerOut.at(j) * (1 - layerOut.at(j));
            }
            else if (activateFunction == "tanh"){
                subCoef *= 1 - pow(tanh(layerOut.at(j)), 2);
            }
            else if (activateFunction == "ReLU"){
                if (layerOut.at(j) <= 0)
                    subCoef = 0;
            }
            else if (activateFunction == "LeakyReLU"){
                if (layerOut.at(j) <= 0)
                    subCoef *= 0.01;
            }
            coef.push_back(subCoef);
            //renew b
            renewWeight.at(row).at(j) -= learningRate * subCoef;
            for (int i = 0; i < row; i++){
                //renew weights
                renewWeight.at(i).at(j) -= learningRate * subCoef * renewLayerOut.at(i);
            }            
        }
        layerVector[layerNo].setCoef(coef);
        layerVector[layerNo - 1].setWeight(renewWeight);
        currentIteration += 1;
    }
    else{
        Layer nextLayer = layerVector.at(layerNo + 1);
        vector<float> nextCoef = nextLayer.getCoef();
        vector<vector<float>> weight = layer.getWeight();
        int nextColumn = nextLayer.getNodeNum() - 1;
        for (int j = 0; j < column; j++){
            float subCoef = 0;
            for (int k = 0; k < nextColumn; k++){
                subCoef += nextCoef.at(k) * weight.at(j).at(k);
            }
            if (activateFunction == "sigmoid"){
                subCoef *= layerOut.at(j) * (1 - layerOut.at(j));
            }
            else if (activateFunction == "tanh"){
                subCoef *= 1 - pow(tanh(layerOut.at(j)), 2);
            }
            else if (activateFunction == "ReLU"){
                if (layerOut.at(j) <= 0)
                    subCoef = 0;
            }
            else if (activateFunction == "LeakyReLU"){
                if (layerOut.at(j) <= 0)
                    subCoef *= 0.01;
            }
            
            coef.push_back(subCoef);
            //renew b
            renewWeight.at(row).at(j) -= learningRate * subCoef;
            for (int i = 0; i < row; i++){
                //renew weights
                renewWeight.at(i).at(j) -= learningRate * subCoef * renewLayerOut.at(i);
            }
        }
        layerVector[layerNo].setCoef(coef);
        layerVector[layerNo - 1].setWeight(renewWeight);        
    }
}

void Net::train(float learning, int max, string activate, string loss, float d){
    assert(find(activateList.begin(), activateList.end(), activate) != activateList.end());
    assert(find(lossList.begin(), lossList.end(), loss) != lossList.end());
    assert(d > 0 && d < 1);
    
    learningRate = learning;
    maxIteration = max;
    activateFunction = activate;
    lossFunction = loss;
    delta = d;
    cout << "[Overview]" << endl;
    cout << "\tlearning rate: " << learning << endl;
    cout << "\tactivate function: " << activate << endl;
    cout << "\tloss function: " << loss << endl;
    cout << "\texpected output: ";
    for (int i = 0; i < output.size(); i++)
        cout << output[i] << " ";
    cout << endl << endl;
    while(currentIteration < maxIteration){
        for (int i = 0; i < layerVector.size() - 1; i++)
            forwardPropagation(i);
        cout << "output: ";
        for (int i = 0; i < layerVector.back().getOut().size(); i++){
            cout << layerVector.back().getOut().at(i) << " ";
        }
        cout << endl;
        lossComputation();
        cout << endl;
        for (int i = (int)layerVector.size() - 1; i > 0; i--)
            backwardPropagation(i);
    }
    for (int i = 0; i < learningRateUpdate.size(); i++){
        cout << "Update the learning rate at iteration " << learningRateUpdate.at(i) << " with the learning rate of " << learning / pow(2, i + 1) << endl;
    }
    cout << "train output: ";
    for (int i = 0; i < layerVector.back().getOut().size(); i++){
        cout << layerVector.back().getOut().at(i) << " ";
    }
    cout << endl;
}

//void Net::testFindOptimalLearningRate(int layerNo){
//    if (layerNo < 1)
//        return;
//    Layer layer = layerVector.at(layerNo);
//    Layer renewLayer = layerVector.at(layerNo - 1);
//    vector<float> layerOut = layer.getOut();
//    vector<float> renewLayerOut = renewLayer.getOut();
//    vector<vector<float>> renewWeight = renewLayer.getWeight();
//    vector<float> coef;
//    vector<float> denominator;
//    vector<float> numerator;
//    int row = renewLayer.getNodeNum() - 1;
//    int column = layer.getNodeNum() - 1;
//    if (layer.getIsOutputLayer() == true){
//        float deno = 0.0;
//        for (int i = 0; i < row; i++){
//            deno += pow(renewLayerOut.at(i), 2);
//        }
//        deno += 1;
//
//        for (int j = 0; j < column; j++){
//            float subCoef = 1;
//            if (lossFunction == "squared"){
//                subCoef *= -(output.at(j) - layerOut.at(j));
//            }
//            else if (lossFunction == "L2"){
//                subCoef *= -(output.at(j) - layerOut.at(j)) / output.size();
//            }
//            else if (lossFunction == "L1"){
//                if (layerOut.at(j) < output.at(j))
//                    subCoef = -1;
//                else if (layerOut.at(j) > output.at(j))
//                    subCoef = 1;
//                else
//                    subCoef = 0;
//            }
//            if (activateFunction == "sigmoid"){
//                subCoef *= layerOut.at(j) * (1 - layerOut.at(j));
//            }
//            coef.push_back(subCoef);
//
//            float numer = 0.0;
////            float numer = 0.0;
//            for (int i = 0; i < row; i++){
//                numer += renewLayerOut.at(i) * renewWeight.at(i).at(j);
////                numer += pow(renewLayerOut.at(i), 2);
//            }
//            numer += renewWeight.at(row).at(j) - output.at(j);
////            numer += 1;
//            denominator.push_back(deno * coef.at(j));
//            numerator.push_back(numer);
//        }
//
//        deno = 0.0;
//        float numer = 0.0;
//        for (int j = 0; j < column; j++){
//            numer += denominator.at(j) * numerator.at(j);
//            deno += pow(denominator.at(j), 2);
//        }
//        learningRate = numer / deno;
//
//        for (int j = 0; j < column; j++){
//            //renew b
//            renewWeight.at(row).at(j) -= learningRate * coef.at(j);
//            for (int i = 0; i < row; i++){
//                //renew weights
//                renewWeight.at(i).at(j) -= learningRate * coef.at(j) * renewLayerOut.at(i);
//            }
//        }
//
//        layerVector[layerNo].setCoef(coef);
//        layerVector[layerNo - 1].setWeight(renewWeight);
//        currentIteration += 1;
//    }
//    else{
//        Layer nextLayer = layerVector.at(layerNo + 1);
//        vector<float> nextCoef = nextLayer.getCoef();
//        vector<vector<float>> weight = layer.getWeight();
//        int nextColumn = nextLayer.getNodeNum() - 1;
//        for (int j = 0; j < column; j++){
//            float subCoef = 0;
//            for (int k = 0; k < nextColumn; k++){
//                subCoef += nextCoef.at(k) * weight.at(j).at(k);
//            }
//            if (activateFunction == "sigmoid"){
//                subCoef *= layerOut.at(j) * (1 - layerOut.at(j));
//            }
//            coef.push_back(subCoef);
//            //renew b
//            renewWeight.at(row).at(j) -= learningRate * subCoef;
//            for (int i = 0; i < row; i++){
//                //renew weights
//                renewWeight.at(i).at(j) -= learningRate * subCoef * renewLayerOut.at(i);
//            }
//        }
//        layerVector[layerNo].setCoef(coef);
//        layerVector[layerNo - 1].setWeight(renewWeight);
//    }
//}
