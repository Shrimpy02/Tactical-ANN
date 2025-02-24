
// Includes
#include "pch.h"
#include <ANN/Neuron.h>
#include <ANN/ActivationFunction.h>
#include <random>


// -------------------- Public --------------------

Neuron::Neuron(const int& _numInputs)
{
	mBias = GetRandomDouble(-0.5, 0.5);
	for(int i = 0; i < _numInputs; i++)
	{
		mWeights.push_back(GetRandomDouble(-0.5, 0.5));
	}
}

// -------------------- Private --------------------

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

void Neuron::ComputeErrorGradient(const double& _targetOutput, ActivationFunctionType _afType)
{
	double error = _targetOutput - mOutput;
	mErrorGradient = error * ActivationFunction::ExecuteActivationFunctionDerivative(mOutput, _afType);
}

void Neuron::UpdateWeights(const double& _learningRate, const std::vector<double>& _inputs)
{
	for (size_t i = 0; i < mWeights.size(); i++)
	{
		mWeights[i] += _learningRate * mErrorGradient * _inputs[i];
	}
	mBias += _learningRate * mErrorGradient;
}

double Neuron::GetRandomDouble(const double& _lowerBound, const double& _upperBound)
{
	std::random_device randomDevice;
	std::mt19937 gen(randomDevice());
	std::uniform_real_distribution<> distribution(_lowerBound, _upperBound);

	return distribution(gen);
}

