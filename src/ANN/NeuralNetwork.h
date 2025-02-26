#pragma once

// Includes
#include "NetworkLayer.h"

/**
 * \brief NeuralNetwork
 * The NeuralNetwork class contains the logic and data for managing a complete neural network comprised of layers, neurons
 * learning algorithms and structure.
 */
 class NeuralNetwork
{
	// -------------------- Variables --------------------
public:

private:

	double mLearningRate;						// The learning rate of the ANN
	std::vector<NetworkLayer> mNetworkLayers;	// Number of layers the network contains, (not considering input layer)

	// -------------------- Functions --------------------
public:

	// Constructs the Artificial Neural Network, input is a vector of pairs where each pair represents a layer in the NN,
	// the first element of the pair is the number of neurons the layer should have, and the second is the activation function for that layer.
	// Important that the first layer contains as many neurons as inputs for the data sett, and the last layer has as many neurons as the desired output data. 
	ANN_API NeuralNetwork(std::vector<std::pair<int, ActivationFunctionType>> _layerCreation, const double& _learningRate = 0.06);

	ANN_API ~NeuralNetwork() = default;

	// Returns a vector of doubles being the ANN`s calculated output based on its input
	ANN_API std::vector<double> CalculateNetworkOutput(const std::vector<double>& _inputs);

	// Trains the ANN, collectively updating weights and biases in reguard to the training input, and the expected outcome,
	// using for the number of epochs as input and with the given learning rate. 
	ANN_API void Train(const std::vector<std::vector<double>>& _trainingInputs,
				const std::vector<std::vector<double>>& _trainingExpectedOutputs,
				const int& _epochs, bool _shouldPrint = false);

private:

	// Back propagation takes the target outputs and
	// updates error gradients for each layer from back to front
	ANN_API void Backpropagate(const std::vector<double>& _expectedOutputs);

	// This function takes in the learning rate and updates the networks neurons weights
	// from front to back, using the error gradient that was calculated from the back propagation
	// times input learning rate and previous layers (IE, the layer in front of the current one) output.
	// the first layer, or input, does not have a weight, there fore the first hidden layers weight update is the input.
	ANN_API void UpdateNetworkWeights(const std::vector<double>& _trainingInputs);
};