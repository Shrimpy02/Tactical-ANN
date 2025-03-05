#pragma once

// Includes
#include "DQN-NetworkLayer.h"
#include <deque>
#include <random>

/**
 * \brief Replay
 * contains the elements to remember like action inputs and so on.
 */
struct Replay
{
	std::vector<double> mInputState;
	int mActionTaken;
	double mRewardReceived;
	std::vector<double> mNextState;
	bool mComplete;
};

/**
 * \brief DQN_ReplayBuffer
 * is a struct that contains previous experiences in a FIFO manor
 */
 struct DQN_ReplayBuffer
{
private:
	std::deque<Replay> mBuffer;
	size_t mCapacity;
	std::mt19937 mRandomEngine;

public:
	DQN_ReplayBuffer(size_t _capacity)
		:	mCapacity(_capacity)
	{
		std::random_device rd;
		mRandomEngine = std::mt19937(rd());
	}

	void Add(const std::vector<double>& _state, int _action, double _reward,
		const std::vector<double>& _next_state, bool _done)
	{
		if (mBuffer.size() >= mCapacity)
			mBuffer.pop_front();

		mBuffer.push_back({ _state, _action, _reward, _next_state, _done });
	}

	std::vector<Replay> Sample(size_t batch_size)
	{
		std::vector<Replay> batch;
		batch.reserve(batch_size);

		std::uniform_int_distribution<size_t> dist(0, mBuffer.size() - 1);

		for (size_t i = 0; i < batch_size; ++i)
		{
			batch.push_back(mBuffer[dist(mRandomEngine)]);
		}

		return batch;
	}

	bool CanSample(size_t batch_size) const
	{
		return mBuffer.size() >= batch_size;
	}

	size_t Size() const
	{
		return mBuffer.size();
	}
};

/**
 * \brief NeuralNetwork
 * The NeuralNetwork class contains the logic and data for managing a complete neural network comprised of layers, neurons
 * learning algorithms and structure.
 */
 class DQN_NeuralNetwork
{
	// -------------------- Variables --------------------
public:

private:

	double mLearningRate;							// The learning rate of the DQN_ANN
	double mGamma;									// The Discount factor
	double mEpsilon;								// The exploration rate
	double mEpsilonDecay;							// The exploration rate falloff (transition to action rate)
	double mMinEpsilon;								// The minimum exploration rate
	DQN_ReplayBuffer mReplayBuffer;					// A replay buffer of past experiences
 	std::vector<DQN_NetworkLayer> mNetworkLayers;	// Number of layers the network contains, (not considering input layer)

	// -------------------- Functions --------------------
public:

	// Default constructor
	ANN_API DQN_NeuralNetwork() = default;

	// Constructs the Artificial Neural Network using Deep Q-Reinforcement learning, input is a vector of pairs where each pair represents a layer in the NN,
	// the first element of the pair is the number of neurons the layer should have. The first layer should contain as many neurons as input states, and the last layer as many outputs as actions.
	// The second is the activation function for that layer. The output layer should be of linear activation type.
	ANN_API DQN_NeuralNetwork(std::vector<std::pair<int, ActivationFunctionType>> _layerCreation, const double& _learningRate = 0.06);

	ANN_API ~DQN_NeuralNetwork() = default;

	// Returns a vector of doubles being the ANN`s calculated output based on its input
	ANN_API std::vector<double> CalculateNetworkOutput(const std::vector<double>& _inputs);

	ANN_API int SelectAction(const std::vector<double>& _inputState);

	ANN_API double CalculateReward();

 	// Trains the ANN, collectively updating weights and biases in reguard to the training input, and the expected outcome,
	// using for the number of epochs as input and with the given learning rate. 
	ANN_API void Train(const std::vector<std::vector<double>>& _trainingInputs,
				const std::vector<std::vector<double>>& _trainingExpectedOutputs,
				const int& _epochs, bool _shouldPrint = false);

	ANN_API void TrainFromReplayBuffer(size_t _batchSize);

	ANN_API void AddExperience(const std::vector<double>& currentState, int action, double reward, const std::vector<double>& nextState, bool done);

private:

	// Back propagation takes the target outputs and
	// updates error gradients for each layer from back to front
	ANN_API void Backpropagate(const std::vector<double>& _expectedOutputs);

	// This function takes in the learning rate and updates the networks neurons weights
	// from front to back, using the error gradient that was calculated from the back propagation
	// times input learning rate and previous layers (IE, the layer in front of the current one) output.
	// the first layer, or input, does not have a weight, there fore the first hidden layers weight update is the input.
	ANN_API void UpdateNetworkWeights(const std::vector<double>& _trainingInputs);


	// Gets a random double between lower bound and upper bound
	ANN_API double GetRandomDouble(const double& _lowerBound, const double& _upperBound);

};