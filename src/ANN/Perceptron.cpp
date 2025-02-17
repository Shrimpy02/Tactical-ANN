// Perceptron.cpp : Defines the exported functions for the DLL.

// Includes
#include "pch.h"
#include "Perceptron.h"

// Additional includes
#include <random>
#include <iostream>

Perceptron::Perceptron(std::vector<LogicSet> _logicTrainingSets)
	: mLogicTrainingSets(_logicTrainingSets)
{
	InitializeRandomWeightsAndBias();
}


void Perceptron::Train(bool _shouldPrint)
{
	uint8_t it = 0;
	for (LogicSet logicSet : mLogicTrainingSets)
	{
		if (_shouldPrint)
		{
			it++;
			std::cout << "LogicSet" << it << ": " << "input1: " << logicSet.mInput.first << "\n";
			std::cout << "LogicSet" << it << ": " << "input2: " << logicSet.mInput.second << "\n";
			std::cout << "Weight1: " << mWeights.first << "\n";
			std::cout << "Weight2: " << mWeights.second << "\n";
			std::cout << "Bias: " << mBias << "\n";
		}

		int calculatedOutput = CalculateOutput(logicSet.mInput.first, logicSet.mInput.second);
		double ErrorDifference = logicSet.mDesiredOutput - calculatedOutput;

		if (_shouldPrint)
		{
			std::cout << "Calculated Output: " << calculatedOutput << "\n";
			std::cout << "Error Difference: " << ErrorDifference << "\n";
			std::cout << "\n";
		}

		mWeights.first = mWeights.first + ErrorDifference * logicSet.mInput.first;
		mWeights.second = mWeights.second + ErrorDifference * logicSet.mInput.second;
		mBias = mBias + ErrorDifference;
	}
}

void Perceptron::Train(const uint16_t& _epochs, bool _shouldPrint)
{
	for (int i = 0; i < _epochs; i++)
	{
		if (_shouldPrint)
			std::cout << "----- Training session " << i + 1 << " ----- \n";
		Train(_shouldPrint);
	}
}

int Perceptron::CalculateOutput(const int& _input1, const int& _input2)
{
	double calculatedOutput = ((_input1 * mWeights.first) + (_input2 * mWeights.second) + mBias);
	calculatedOutput = calculatedOutput > 0 ? 1 : 0;
	return static_cast<int>(calculatedOutput);
}

void Perceptron::InitializeRandomWeightsAndBias()
{
	mWeights.first = GetRandomDouble(-1, 1);
	mWeights.second = GetRandomDouble(-1, 1);
	mBias = GetRandomDouble(-1, 1);
}

double Perceptron::GetRandomDouble(const double& _lowerBound, const double& _upperBound)
{
	std::random_device randomDevice;
	std::mt19937 gen(randomDevice());
	std::uniform_real_distribution<> distribution(_lowerBound, _upperBound);

	return distribution(gen);
}

