#pragma once

// Includes
#include <ANN/Neuron.h>
#include <vector>
#include <memory>

/**
 * \brief Network layer
 * The network layer class manages all individual neurons in each layer of an artificial neural network.
 * It contains the functions to calculate collective output, error gradient and weight updates. 
 */
class NetworkLayer
{
	// -------------------- Variables --------------------
public:

	std::vector<std::shared_ptr<Neuron>> mLayerNeurons; // all neurons in layer
	std::vector<double> mLayerOutputs;					// Collective layer output

private:

	// -------------------- Functions --------------------
public:

	// Construct the network layer with the number of neurons and the number of inputs each neuron should have
	NetworkLayer(const int& _numNeurons, const int& _numInputsPerNeuron);

	~NetworkLayer() = default;

	// Calculates the collective output from each neuron in this layer, and stores it
	std::vector<double> CalculateLayerOutput(const std::vector<double>& _inputs, ActivationFunctionType _aft);

	// Computes and updates each neuron in the layers error gradient 
	void ComputeErrorGradient(ActivationFunctionType _aft);

	// Updates each neuron in the layers weight based on the input
	void UpdateWeights(const double& _learningRate, const std::vector<double>& inputs);

private:


};