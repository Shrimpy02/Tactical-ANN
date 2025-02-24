#pragma once

// Includes
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
	double mBias;			// Bias value for the neuron, used in its activation function
	double mNetInput;		// The net input is the value before activation function is applied
	double mOutput;			// The output value of the neuron after processing inputs  and activation function
	double mErrorGradient;	// The gradient of the error for this neuron, used during back propagation

	std::vector<double> mWeights; // Dynamic array of weights for each input

	// -------------------- Functions --------------------
public:

	// Constructs the neuron by randomizing weights equal to the amount of inputs it can expect.
	 Neuron(const int& _numInputs);

	~Neuron() = default;

	// Calculates the output from the given inputs, using the given activation function type, using this neurons weight and bias values. 
	double CalculateOutput(const std::vector<double>& _inputs, ActivationFunctionType _afType);

	// Computes and updates the neurons error gradient
	void ComputeErrorGradient(const double& _targetOutput, ActivationFunctionType _afType);

	// Updates the weights of the neuron by learning rate, input and local error gradient
	void UpdateWeights(const double& _learningRate, const std::vector<double>& _inputs);

private:
	// Gets a random double between lower bound and upper bound
	double GetRandomDouble(const double& _lowerBound, const double& _upperBound);

};