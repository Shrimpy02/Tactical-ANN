
// Includes
#include "pch.h"
#include "DQN-NeuralNetwork.h"
#include <iostream>

// -------------------- Public --------------------

DQN_NeuralNetwork::DQN_NeuralNetwork(std::vector<std::pair<int, ActivationFunctionType>> _layerCreation, const double& _learningRate)
	:	mLearningRate(_learningRate), mGamma(0.99), mEpsilon(1.0), mEpsilonDecay(0.995),  mMinEpsilon(0.01), mReplayBuffer(1000)
{
	// skips the first layer since it is regarded as the inputs and does therefore not need weights or biases.
	for(int i = 1; i < _layerCreation.size(); i++)
		mNetworkLayers.emplace_back(DQN_NetworkLayer(_layerCreation[i].first, _layerCreation[i].second, _layerCreation[i-1].first));
}

std::vector<double> DQN_NeuralNetwork::CalculateNetworkOutput(const std::vector<double>& _inputs)
{
	std::vector<double> currentOutputs = _inputs;

	for (DQN_NetworkLayer& layer : mNetworkLayers)
		currentOutputs = layer.CalculateLayerOutput(currentOutputs);

	// Final output of the network
	return currentOutputs;  
}

int DQN_NeuralNetwork::SelectAction(const std::vector<double>& _inputState)
{
	if (GetRandomDouble(0.0, 1.0) < mEpsilon) {
		return rand() % mNetworkLayers.back().GetOutput().size();
	}
	else {
		std::vector<double> qValues = CalculateNetworkOutput(_inputState);
		return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
	}
}

double DQN_NeuralNetwork::CalculateReward()
{
	return 0;
}

void DQN_NeuralNetwork::Train(const std::vector<std::vector<double>>& _inputState,
                              const std::vector<std::vector<double>>& _expectedOutputState,
                              const int& _epochs, bool _shouldPrint)
{
	for (int epoch = 0; epoch < _epochs; epoch++)
	{
		
		// Sample a batch from the replay buffer
		if (mReplayBuffer.CanSample(10))
		{
			// Train the network from this sampled batch
			TrainFromReplayBuffer(10);
		}

		double totalError = 0.0;
		for (size_t i = 0; i < _inputState.size(); i++)
		{
			if (i + 1 >= _inputState.size())
				continue;

			// Select Action by Q-Value
			int action = SelectAction(_inputState[i]);

			double reward = CalculateReward();

			std::vector<double> nextState = _inputState[i + 1];

			// Add experience to the replay buffer (for experience replay)
			AddExperience(_inputState[i], action, reward, nextState, false);

			// Compute the predicted Q-values for the current state
			std::vector<double> predictedQValues = CalculateNetworkOutput(_inputState[i]);

			// Compute the target Q-value using the reward and max Q-value from the next state
			double maxNextQValue = *std::max_element(CalculateNetworkOutput(nextState).begin(), CalculateNetworkOutput(nextState).end());
			double targetQValue = reward + mGamma * maxNextQValue;  // mGamma is the discount factor

			// Calculate the TD error (difference between predicted and target Q-value)
			double tdError = predictedQValues[action] - targetQValue;

			// Compute the loss (Mean Squared Error between the predicted Q-value and target Q-value)
			totalError += tdError * tdError;

			// Back propagation
			Backpropagate(_expectedOutputState[i]);

			// Update Weights
			UpdateNetworkWeights(_inputState[i]);
		}

		// print epoch and error gradient:
		if(_shouldPrint)
			std::cout << "Epoch " << epoch + 1 << " - Error: " << totalError / _inputState.size() << std::endl;
	}
}

void DQN_NeuralNetwork::TrainFromReplayBuffer(size_t _batchSize)
{
	if (mReplayBuffer.Size() < _batchSize)
		return;

	std::vector<Replay> batch = mReplayBuffer.Sample(_batchSize);

	for (const Replay& experience : batch) {
		std::vector<double> currentQValues = CalculateNetworkOutput(experience.mInputState);

		std::vector<double> nextQValues = CalculateNetworkOutput(experience.mNextState);
		double maxNextQ = *std::max_element(nextQValues.begin(), nextQValues.end());

		double targetQ = experience.mRewardReceived;
		if (!experience.mComplete)
			targetQ += mGamma * maxNextQ;

		std::vector<double> targetOutputs = currentQValues;
		targetOutputs[experience.mActionTaken] = targetQ; // Update only the taken action

		Backpropagate(targetOutputs);
		UpdateNetworkWeights(experience.mInputState);
	}
}

void DQN_NeuralNetwork::AddExperience(const std::vector<double>& currentState, int action, double reward, const std::vector<double>& nextState, bool done)
{
	mReplayBuffer.Add(currentState, action, reward, nextState, done);

	// Decay epsilon for exploration-exploitation balance
	if (mEpsilon > mMinEpsilon)
	{
		mEpsilon *= mEpsilonDecay;
	}
}

// -------------------- Private --------------------


void DQN_NeuralNetwork::Backpropagate(const std::vector<double>& _expectedOutputs)
{
	// Update output layer with target output
	mNetworkLayers.back().ComputeErrorGradientLayer(_expectedOutputs);

	// propagate back to front for hidden layers
	for (int i = static_cast<int>(mNetworkLayers.size() - 2); i >= 0; i--)
		mNetworkLayers[i].ComputeErrorGradientLayer(mNetworkLayers[i + 1]);
}

void DQN_NeuralNetwork::UpdateNetworkWeights(const std::vector<double>& _trainingInputs)
{
	// First hidden layer should use training inputs
	std::vector<double> previousOutputs = _trainingInputs;

	// the rest of the layers should use successive output
	for (DQN_NetworkLayer& layer : mNetworkLayers)
	{
		layer.UpdateLayerWeights(mLearningRate, previousOutputs);
		previousOutputs = layer.GetOutput();
	}
	
}

double DQN_NeuralNetwork::GetRandomDouble(const double& _lowerBound, const double& _upperBound)
{
	std::random_device randomDevice;
	std::mt19937 gen(randomDevice());
	std::uniform_real_distribution<> distribution(_lowerBound, _upperBound);

	return distribution(gen);
}
