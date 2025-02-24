
// Includes
#include "pch.h"
#include <ANN/NetworkLayer.h>

// -------------------- Public --------------------

NetworkLayer::NetworkLayer(const int& _numNeurons, const int& _numInputsPerNeuron)
{
	for(int i = 0; i < _numNeurons; i++)
	{
		mLayerNeurons.push_back(std::make_shared<Neuron>(_numInputsPerNeuron));
	}
}

std::vector<double> NetworkLayer::CalculateLayerOutput(const std::vector<double>& _inputs, ActivationFunctionType _aft)
{
	mLayerOutputs.clear();

	for(std::shared_ptr<Neuron> neuron : mLayerNeurons)
		mLayerOutputs.push_back(neuron->CalculateOutput(_inputs, _aft));

	return mLayerOutputs;
}

void NetworkLayer::ComputeErrorGradient(ActivationFunctionType _aft)
{
	for (size_t i = 0; i < mLayerNeurons.size(); i++)
	{
		mLayerNeurons[i]->CalculateErrorGradient(mLayerOutputs[i], _aft);
	}
}

void NetworkLayer::UpdateWeights(const double& _learningRate, const std::vector<double>& inputs)
{
	for (std::shared_ptr<Neuron> neuron : mLayerNeurons)
		neuron->UpdateWeights(_learningRate, inputs);
}

// -------------------- Private --------------------
