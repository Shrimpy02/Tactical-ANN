#pragma once

// Includes
#include <ANN/NetworkLayer.h>
#include <ANN/ActivationFunction.h>
#include <vector>

class NeuralNetwork
{
	// -------------------- Variables --------------------
public:

private:

	int mNumInputs;
	int mNumOutputs;
	int mNumHiddenLayers;
	int mNumHiddenNeurons;
	double mLearningRate;
	std::vector<NetworkLayer> mNetworkLayers;

	// -------------------- Functions --------------------
public:

	NeuralNetwork(const int& _numInputs, const int& _numOutputs, const int& _numHiddenLayers,
		const int& _numHiddenNeurons, const double& _learningRate, ActivationFunctionType _hiddenAF
		, ActivationFunctionType _outputAF);

	~NeuralNetwork() = default;

private:

};