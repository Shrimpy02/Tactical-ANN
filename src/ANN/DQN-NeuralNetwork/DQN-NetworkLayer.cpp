
// Includes
#include "pch.h"
#include "DQN-NetworkLayer.h"
#include <random>

// -------------------- Public --------------------

DQN_NetworkLayer::DQN_NetworkLayer(const int& _numNeurons, ActivationFunctionType _aft, const int& _numExpectedInputs)
	:	mLayerActivationFunctionType(_aft)
{
	for(int i = 0; i < _numNeurons; i++)
	{
		std::vector<double> randomWeights(_numExpectedInputs);
		for (int j = 0; j < _numExpectedInputs; j++)
			randomWeights[j] = GetRandomDouble(-0.5, 0.5);

		double randomBias = GetRandomDouble(-0.5, 0.5);

		mLayerNeurons.push_back(std::make_shared<DQN_Neuron>(randomBias, randomWeights));
	}
}

std::vector<double> DQN_NetworkLayer::CalculateLayerOutput(const std::vector<double>& _inputs)
{
	if (mLayerOutputs.size() != mLayerNeurons.size())
		mLayerOutputs.resize(mLayerNeurons.size());

	for (size_t i = 0; i < mLayerNeurons.size(); i++)
		mLayerOutputs[i] = mLayerNeurons[i]->CalculateOutput(_inputs, mLayerActivationFunctionType);

	return mLayerOutputs;
}

void DQN_NetworkLayer::ComputeErrorGradientLayer(const DQN_NetworkLayer& _nextLayer)
{
	for (int i = 0; i < static_cast<int>(mLayerNeurons.size()); i++)
	{
		double sumWeightedGradients = 0.0;

		// Sum the error gradient contributions from the next layer
		for (const std::shared_ptr<DQN_Neuron>& neuron: _nextLayer.GetNeurons())
			sumWeightedGradients += neuron->GetWeight(i) * neuron->GetErrorGradient();

		// Apply activation function derivative on neuron itself
		mLayerNeurons[i]->ComputeErrorGradient(sumWeightedGradients, mLayerActivationFunctionType);
	}
}

void DQN_NetworkLayer::ComputeErrorGradientLayer(const std::vector<double>& _expectedOutput)
{
	for (size_t i = 0; i < mLayerNeurons.size(); i++)
		mLayerNeurons[i]->ComputeErrorGradient(_expectedOutput[i] - mLayerNeurons[i]->GetOutput(), mLayerActivationFunctionType);
}

void DQN_NetworkLayer::UpdateLayerWeights(const double& _learningRate, const std::vector<double>& _previousOutput)
{
	for (std::shared_ptr<DQN_Neuron>& neuron : mLayerNeurons)
		neuron->UpdateWeights(_learningRate, _previousOutput);
}

std::vector<double> DQN_NetworkLayer::GetLayerErrorGradient()
{
	std::vector<double> totalErrorGradient;

	for (std::shared_ptr<DQN_Neuron> neuron : mLayerNeurons)
		totalErrorGradient.push_back(neuron->GetErrorGradient());

	return totalErrorGradient;
}

std::vector<double> DQN_NetworkLayer::GetOutput()
{
	std::vector<double> outputs;

	for (const std::shared_ptr<DQN_Neuron>& neuron : mLayerNeurons)
		outputs.push_back(neuron->GetOutput());

	return outputs;
}

const std::vector<std::shared_ptr<DQN_Neuron>>& DQN_NetworkLayer::GetNeurons() const
{
	return mLayerNeurons;
}

// -------------------- Private --------------------

double DQN_NetworkLayer::GetRandomDouble(const double& _lowerBound, const double& _upperBound)
{
	std::random_device randomDevice;
	std::mt19937 gen(randomDevice());
	std::uniform_real_distribution<> distribution(_lowerBound, _upperBound);

	return distribution(gen);
}
