//
//  Layer.hpp
//  test1118
//
//  Created by 许清嘉 on 11/18/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <iostream>
#include <vector>

using namespace std;

class Layer{
private:
    vector<vector<float>> weight;
    vector<float> out;
    int nodeNum;
    vector<float> coef;//common product item used to update weight
    bool isOutputLayer;
public:
    Layer();
    Layer(int);
//    Layer(vector<vector<float>>, int);
    void setWeight(vector<vector<float>>&);
    vector<vector<float>>& getWeight();
    void setOut(vector<float>&);
    vector<float>& getOut();
    void setNodeNum(int);
    int getNodeNum() const;
    void setCoef(vector<float>&);
    vector<float>& getCoef();
    void setIsOutputLayer(bool);
    bool getIsOutputLayer() const;
};

#endif /* Layer_hpp */
