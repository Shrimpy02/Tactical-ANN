#pragma once

// Includes
#include "DQN-Neuron.h"

/**
 * \brief Network layer
 * The network layer class manages all individual neurons in each layer of an artificial neural network.
 * It contains the functions to calculate collective output, error gradient and weight updates. 
 */
class DQN_NetworkLayer
{
	// -------------------- Variables --------------------
public:

private:

	std::vector<std::shared_ptr<DQN_Neuron>> mLayerNeurons;		// all neurons in layer
	std::vector<double> mLayerOutputs;						// Collective layer output
	ActivationFunctionType mLayerActivationFunctionType;	// the activation function this layer uses

	// -------------------- Functions --------------------
public:

	// Construct the network layer with the number of neurons and the number of inputs each neuron should have
	ANN_API DQN_NetworkLayer(const int& _numNeurons, ActivationFunctionType _aft, const int& _numExpectedInputs);

	ANN_API ~DQN_NetworkLayer() = default;

	// Calculates the collective output from each neuron in this layer, and stores it
	ANN_API std::vector<double> CalculateLayerOutput(const std::vector<double>& _inputs);

	// Overloadable, Computes and updates each neuron in the layers error gradient back to front
	// by passing the last layer processed, usable for hidden layer, while output layer requires expected output
	ANN_API void ComputeErrorGradientLayer(const DQN_NetworkLayer& _nextLayer);

	// Overloadable, Computes and updates each neuron in the layers error gradient
	// taking in the expected output so the output layer can update their error gradient so
	// it can be passed into the hidden layers.
	ANN_API void ComputeErrorGradientLayer(const std::vector<double>& _expectedOutput);

	// Updates each neuron in the layers weight based on the input learning rate, and the previous layers ouput
	ANN_API void UpdateLayerWeights(const double& _learningRate, const std::vector<double>& _previousOutput);

	// Returns the previous calculated output of all neurons as a vector of doubles
	ANN_API std::vector<double> GetOutput();

private:

	// Returns a pointer to the vector of pointers to the layers neurons
	ANN_API const std::vector<std::shared_ptr<DQN_Neuron>>& GetNeurons() const;

	// Gets the total error gradient of all neurons in this layer as a vector of doubles
	ANN_API std::vector<double> GetLayerErrorGradient();

	// Gets a random double between lower bound and upper bound
	ANN_API double GetRandomDouble(const double& _lowerBound, const double& _upperBound);

};