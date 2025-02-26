// Perceptron.h - Contains declarations of perceptron functions
#pragma once

#include "Defines.h"
#include <utility>
#include <vector>
#include <string>

struct LogicSet
{
    std::pair<int, int> mInput;    // Logic input values
    int mDesiredOutput;               // Expected input value result

    ANN_API  LogicSet(int _i1, int _i2, int _desiredOutput)
        : mInput(std::make_pair(_i1, _i2)), mDesiredOutput(_desiredOutput)
    {
    }

};

class Perceptron
{
public:
    // ---- Variables --------

    // ---- Functions -----

    // Constructor, initializes training set from input and random weights and bias.
    ANN_API  Perceptron(std::vector<LogicSet> _logicTrainingSets);

    // Default de-constructor
    ANN_API  ~Perceptron() = default;

    // Trains a single iteration of all logic configurations
    ANN_API  void Train(bool _shouldPrint);

    // Trains the Perceptron equal times to the number of epochs.
    ANN_API  void Train(const uint16_t& _epochs, bool _shouldPrint);

    // Calculates the result of the two input int`s, this is the Activation function
    ANN_API  int CalculateOutput(const int& _input1, const int& _input2);

private:
    // ---- Variables --------

    std::vector<LogicSet> mLogicTrainingSets;   // The training set
    std::pair<double, double> mWeights = { 0,0 }; // The weight
    double mBias = 0;                           // The weight bias
    double mTotalError = 0;                     // The total error

    // ---- Functions -----

    // Initialize random weights and biases for this Perceptron
    ANN_API  void InitializeRandomWeightsAndBias();

    // Returns a random double between the lower and upper bounds input.
    ANN_API  double GetRandomDouble(const double& _lowerBound, const double& _upperBound);


};
