//
//  Layer.cpp
//  test1118
//
//  Created by 许清嘉 on 11/18/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "Layer.hpp"

Layer::Layer(){
    
}

Layer::Layer(int nodeNum){
    this->nodeNum = nodeNum + 1;
    isOutputLayer = false;
}

void Layer::setWeight(vector<vector<float>>& newWeight){
    weight = newWeight;
}

vector<vector<float>>& Layer::getWeight(){
    return weight;
}

void Layer::setOut(vector<float>& out){
    this->out = out;
    if (!isOutputLayer)
        this->out.push_back(1);
}

vector<float>& Layer::getOut(){
    return out;
}

void Layer::setNodeNum(int newNodeNum){
    nodeNum = newNodeNum + 1;
}

int Layer::getNodeNum() const{
    return nodeNum;
}

void Layer::setCoef(vector<float>& newCoef){
    coef = newCoef;
}

vector<float>& Layer::getCoef(){
    return coef;
}

void Layer::setIsOutputLayer(bool newIsOutputLayer){
    isOutputLayer = newIsOutputLayer;
}

bool Layer::getIsOutputLayer() const{
    return isOutputLayer;
}
