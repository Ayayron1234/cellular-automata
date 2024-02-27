#pragma once
#include "../device_helpers.h"

using Neuron = float;
using Weight = float;

class Layer {
public:
	virtual Neuron Get(int index) const = 0;

	virtual int NeuronCount() const = 0;

protected:
	int m_neuronCount;
	GlobalReadWriteBuffer<Neuron> m_neurons;

};

class HiddenLayer : public Layer {
	HiddenLayer(int neuronCount, const Layer& previosLayer)
	{
		m_neuronCount = neuronCount;
		m_neurons.SetUp(neuronCount);
		m_weights.SetUp(neuronCount * previosLayer.NeuronCount());
		m_biases.SetUp(neuronCount);
	}

	const Neuron& ReadNeuron(int index) const {
		return m_neurons.Read(index);
	}

	const Weight& ReadWeight(int index) const {
		return m_weights.Read(index);
	}

	const Weight& ReadBias(int index) const {
		return m_biases.Read(index);
	}

	void SetWeight(int index, const Weight& value) {
		m_weights.Write(index, value);
	}

	void SetBias(int index, const Weight& value) {
		m_biases.Write(index, value);
	}

	int NeuronCount() const {
		return m_neuronCount;
	}

private:
	GlobalReadWriteBuffer<Weight> m_weights;
	GlobalReadWriteBuffer<Weight> m_biases;

};

class InputLayer : public Layer {
public:
	in
};

using OutputLayer = HiddenLayer;
