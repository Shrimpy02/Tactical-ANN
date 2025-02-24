
// Includes
#include "pch.h"
#include <ANN/NeuralNetwork.h>

// -------------------- Public --------------------

NeuralNetwork::NeuralNetwork(const int& _numInputs, const int& _numOutputs, const int& _numHiddenLayers,
							const int& _numHiddenNeurons, const double& _learningRate, ActivationFunction::ActivationFunctionType _hiddenAF
							, ActivationFunction::ActivationFunctionType _outputAF)
: mNumInputs(_numInputs), mNumOutputs(_numOutputs), mNumHiddenLayers(_numHiddenLayers),
mNumHiddenNeurons(_numHiddenNeurons), mLearningRate(_learningRate)
{

}

// -------------------- Private --------------------
