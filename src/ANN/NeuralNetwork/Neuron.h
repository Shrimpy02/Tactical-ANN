#pragma once

// Includes
#include "ANN/ActivationFunction.h"
#include <vector>

// Forward declare
enum ActivationFunctionType;

/**
 * \brief Neuron
 * The Neuron class manages individual properties of an artificial neural network`s neurons. It manages
 * initialization, calculation output based on activation function and updates bias and weights. 
*/
 class Neuron
{
	// -------------------- Variables --------------------
public:

private:
	double mBias;				// Bias value for the neuron, used in its activation function
	double mNetInput = 0;			// The net input is the value before activation function is applied
	double mOutput = 0;				// The output value of the neuron after processing inputs  and activation function
	double mErrorGradient = 0;	// The gradient of the error for this neuron, used during back propagation

	std::vector<double> mWeights; // Dynamic array of weights for each input

	// -------------------- Functions --------------------
public:

	// Constructs the neuron by randomizing weights equal to the amount of inputs it can expect.
	ANN_API Neuron(const double& _initialBias, std::vector<double>& _initialWeight);

	ANN_API ~Neuron() = default;

	// Calculates the output from the given inputs, using the given activation function type, using this neurons weight and bias values. 
	ANN_API double CalculateOutput(const std::vector<double>& _inputs, ActivationFunctionType _afType);

	// Computes and updates the neurons error gradient with input error signal using input activation function type
	ANN_API void ComputeErrorGradient(const double& _errorSignal, ActivationFunctionType _afType);

	// Updates the weights of the neuron by learning rate, previous layers Output and local error gradient
	ANN_API void UpdateWeights(const double& _learningRate, const std::vector<double>& _previousOutput);

	// Returns the error gradient of a neuron as a double. 
	ANN_API double GetErrorGradient();

	// Returns the output of the neuron as a double.  
	ANN_API double GetOutput();

	// Returns a single weight of the neuron given the input index
	ANN_API double GetWeight(const int& _index);

private:

};