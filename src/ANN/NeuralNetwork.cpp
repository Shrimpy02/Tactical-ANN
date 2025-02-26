
// Includes
#include "pch.h"
#include "NeuralNetwork.h"
#include <iostream>

// -------------------- Public --------------------

NeuralNetwork::NeuralNetwork(std::vector<std::pair<int, ActivationFunctionType>> _layerCreation, const double& _learningRate)
	:	mLearningRate(_learningRate)
{
	// skips the first layer since it is regarded as the inputs and does therefore not need weights or biases.
	for(int i = 1; i < _layerCreation.size(); i++)
		mNetworkLayers.emplace_back(NetworkLayer(_layerCreation[i].first, _layerCreation[i].second, _layerCreation[i-1].first));
}

std::vector<double> NeuralNetwork::CalculateNetworkOutput(const std::vector<double>& _inputs)
{
	std::vector<double> currentOutputs = _inputs;

	for (NetworkLayer& layer : mNetworkLayers)
		currentOutputs = layer.CalculateLayerOutput(currentOutputs);

	// Final output of the network
	return currentOutputs;  
}

void NeuralNetwork::Train(const std::vector<std::vector<double>>& _trainingInputs,
							const std::vector<std::vector<double>>& _trainingExpectedOutputs,
								const int& _epochs, bool _shouldPrint)
{
	for (int epoch = 0; epoch < _epochs; epoch++)
	{
		double totalError = 0.0;
		for (size_t i = 0; i < _trainingInputs.size(); i++)
		{
			// Calculate output
			std::vector<double> outputs = CalculateNetworkOutput(_trainingInputs[i]);

			// Compute error (Mean Squared Error)
			double epochError = 0.0;
			for (size_t j = 0; j < outputs.size(); j++) {
				double error = _trainingExpectedOutputs[i][j] - outputs[j];
				epochError += error * error;
			}
			totalError += epochError;

			// Back propagation
			Backpropagate(_trainingExpectedOutputs[i]);

			// Update Weights
			UpdateNetworkWeights(_trainingInputs[i]);
		}

		// print epoch and error gradient:
		if(_shouldPrint)
			std::cout << "Epoch " << epoch + 1 << " - Error: " << totalError / _trainingInputs.size() << std::endl;
	}
}

// -------------------- Private --------------------


void NeuralNetwork::Backpropagate(const std::vector<double>& _expectedOutputs)
{
	// Update output layer with target output
	mNetworkLayers.back().ComputeErrorGradientLayer(_expectedOutputs);

	// propagate back to front for hidden layers
	for (int i = static_cast<int>(mNetworkLayers.size() - 2); i >= 0; i--)
		mNetworkLayers[i].ComputeErrorGradientLayer(mNetworkLayers[i + 1]);
	
}

void NeuralNetwork::UpdateNetworkWeights(const std::vector<double>& _trainingInputs)
{
	// First hidden layer should use training inputs
	std::vector<double> previousOutputs = _trainingInputs;

	// the rest of the layers should use successive output
	for (NetworkLayer& layer : mNetworkLayers)
	{
		layer.UpdateLayerWeights(mLearningRate, previousOutputs);
		previousOutputs = layer.GetOutput();
	}
	
}