
// Includes
#include "pch.h"
#include "Neuron.h"

// -------------------- Public --------------------

Neuron::Neuron(const double& _initialBias, std::vector<double>& _initialWeight)
	:	mBias(_initialBias), mWeights(std::move(_initialWeight))
{}

double Neuron::CalculateOutput(const std::vector<double>& _inputs, ActivationFunctionType _afType)
{
	if (_inputs.size() != mWeights.size()) {
		throw std::invalid_argument("Input size does not match weight size!");
	}

	double output = mBias;
	for(size_t i = 0; i < _inputs.size(); i++)
	{
		output += mWeights[i] * _inputs[i];
	}

	mNetInput = output;

	mOutput = ActivationFunction::ExecuteActivationFunction(output, _afType);

	return mOutput;
}

void Neuron::ComputeErrorGradient(const double& _errorSignal, ActivationFunctionType _afType)
{
	mErrorGradient = _errorSignal * ActivationFunction::ExecuteActivationFunctionDerivative(mNetInput, _afType);
}

void Neuron::UpdateWeights(const double& _learningRate, const std::vector<double>& _previousLayerOutput)
{
	for (size_t i = 0; i < mWeights.size(); i++)
	{
		mWeights[i] += _learningRate * mErrorGradient * _previousLayerOutput[i];
	}
	mBias += _learningRate * mErrorGradient;
}

double Neuron::GetErrorGradient()
{
	return mErrorGradient;
}

double Neuron::GetOutput()
{
	return mOutput;
}

double Neuron::GetWeight(const int& _index)
{
	if(_index >= static_cast<int>(mWeights.size()) || _index < 0) 
		throw std::invalid_argument("Index not within neuron weight bounds");

	return mWeights[_index];
}

// -------------------- Private --------------------
