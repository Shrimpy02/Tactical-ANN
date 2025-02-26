#pragma once

// Includes
#include "Defines.h"
#include <complex>
#include <memory>
#include <utility>

enum ActivationFunctionType
{
	AFT_Identity,
	AFT_BinaryStep,
	AFT_Sigmoid,
	AFT_Hyperbolic,
	AFT_RectifiedLinear,
	AFT_LeakyRectified
};

class ActivationFunction
{
public:

	ANN_API static double ExecuteActivationFunction(double _calculatedOutput, ActivationFunctionType _aft)
	{
		switch (_aft)
		{
		case ActivationFunctionType::AFT_Identity:
			return Identity(_calculatedOutput);
		case ActivationFunctionType::AFT_BinaryStep:
			return BinaryStep(_calculatedOutput);
		case ActivationFunctionType::AFT_Sigmoid:
			return Sigmoid(_calculatedOutput);
		case ActivationFunctionType::AFT_Hyperbolic:
			return Hyperbolic(_calculatedOutput);
		case ActivationFunctionType::AFT_RectifiedLinear:
			return RectifiedLinear(_calculatedOutput);
		case ActivationFunctionType::AFT_LeakyRectified:
			return LeakyRectified(_calculatedOutput);
		}

		throw std::invalid_argument("No Activation function mach");
		return -1;
	}

	ANN_API static double ExecuteActivationFunctionDerivative(double _calculatedOutput, ActivationFunctionType _aft)
	{
		switch (_aft)
		{
		case ActivationFunctionType::AFT_Identity:
			return IdentityDeriv();
		case ActivationFunctionType::AFT_BinaryStep:
			return BinaryStepDeriv();
		case ActivationFunctionType::AFT_Sigmoid:
			return SigmoidDeriv(_calculatedOutput);
		case ActivationFunctionType::AFT_Hyperbolic:
			return HyperbolicDeriv(_calculatedOutput);
		case ActivationFunctionType::AFT_RectifiedLinear:
			return RectifiedLinearDeriv(_calculatedOutput);
		case ActivationFunctionType::AFT_LeakyRectified:
			return LeakyRectifiedDeriv(_calculatedOutput);
		}

		throw std::invalid_argument("No derivative activation function mach");
		return -1;
	}

private:

	ANN_API static double Identity(double _calculatedOutput)
	{
		return _calculatedOutput;
	}
	ANN_API static double IdentityDeriv()
	{
		return 1;
	}

	ANN_API static double BinaryStep(double _calculatedOutput)
	{
		return _calculatedOutput >= 0 ? 1 : 0;
	}
	ANN_API static double BinaryStepDeriv()
	{
		return 0;
	}

	ANN_API static double Sigmoid(double _calculatedOutput)
	{
		return 1.0 / (1.0 + std::exp(-_calculatedOutput));
	}
	ANN_API static double SigmoidDeriv(double _calculatedOutput)
	{
		double sig = Sigmoid(_calculatedOutput);
		return sig * (1 - sig);
	}

	ANN_API static double Hyperbolic(double _calculatedOutput)
	{
		return tanh(_calculatedOutput);
	}
	ANN_API static double HyperbolicDeriv(double _calculatedOutput)
	{
		double Hyper = Hyperbolic(_calculatedOutput);
		return 1 - std::pow(Hyper, 2);

	}

	ANN_API static double RectifiedLinear(double _calculatedOutput)
	{
		return _calculatedOutput > 0 ? _calculatedOutput : 0;
	}
	ANN_API static double RectifiedLinearDeriv(double _calculatedOutput)
	{
		return _calculatedOutput > 0 ? 1 : 0;
	}

	ANN_API static double LeakyRectified(double _calculatedOutput)
	{
		return _calculatedOutput <= 0 ? 0.01 * _calculatedOutput : _calculatedOutput;
	}
	ANN_API static double LeakyRectifiedDeriv(double _calculatedOutput)
	{
		return _calculatedOutput < 0 ? 0.01 : 1;
	}

};