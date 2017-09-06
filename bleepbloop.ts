/*
Copyright 2017
Fredrik August Madsen-Malmo

bleepbloop.ts -- main file that binds the library together
*/

// Helper functions
function randomWeight(): number {
    return Math.random() * .2;
}

function sigmoid(input: number): number {
    return 1 / (1 + Math.pow(Math.E, input));
}

function error(target: number[], output: number[]) {
    return .5 * target.reduce((acc, curr, i) => {
        if (i == 0) {
            return Math.pow(acc - output[i], 2);
        }

        return acc + Math.pow(curr - output[i], 2);
    });
}

class Neuron {
    weights: number[];
    output: number;
    inputs: number[];

    activate(inputs: number[], bias: number) {
        // For the backpropagation
        this.inputs = inputs;

        return inputs.reduce((acc, curr, i) => {
            if (i == 0) {
                return acc * this.weights[i];
            }

            return acc + curr * this.weights[i]
        }) + bias;
    }

    transfer(inputs: number[], bias: number) {
        this.output = sigmoid(this.activate(inputs, bias));

        return this.output;
    }
}

class Layer {
    neurons: Neuron[];
    bias: number;

    constructor(unitCount: number) {
        this.neurons = Array.apply(null, new Array(unitCount)).map(() => (new Neuron));
    }
}

class Network {
    layers: Layer[];

    constructor(public layerLenghts: number[], public learningRate: number) {
        this.layers = layerLenghts.map((layerLength, i) => {
            return new Layer(layerLenghts[i]);
        });

        this.seedWeightsAndBias();
    }

    seedWeightsAndBias() {
        for (let layerIndexStr in this.layers) {
            let layerIndex: number = Number(layerIndexStr);
            if (layerIndex == 0) { continue; }

            let layer: Layer = this.layers[layerIndex];

            layer.bias = randomWeight();

            for (let unit of layer.neurons) {
                unit.weights = Array.apply(null, new Array(this.layers[layerIndex - 1].neurons.length)).map(() => (
                    randomWeight()
                ));
            }
        }
    }

    predict(inputs: number[]): number[] {
        if (inputs.length != this.layers[0].neurons.length) {
            console.error('Input length must be the same as number of input units!');
            return [];
        }

        let prevLayerOutputs: number[] = inputs;

        //console.log(prevLayerOutputs);

        for (let layerIndexStr in this.layers) {
            let layerIndex: number = Number(layerIndexStr);

            if (layerIndex == 0) {
                continue;
            }

            prevLayerOutputs = this.layers[layerIndex].neurons.map((neuron) => (
                neuron.transfer(prevLayerOutputs, this.layers[layerIndex].bias)
            ));
        }

        return prevLayerOutputs;
    }

    train(trainingExamples: [number[]], trainingTargets: [number[]]) {
        // Using backpropagation
        if (trainingExamples[0].length != this.layers[0].neurons.length) {
            console.error('Input length must be the same as number of input units!');
            return;
        }

        if (trainingTargets[0].length != this.layers[this.layers.length - 1].neurons.length) {
            console.error('Target length must be the same as number of utput units!');
            return;
        }

        if (trainingExamples.length != trainingTargets.length) {
            console.error('Examples must be of same length as targets!');
            return;
        }

        for (let trainingExampleIndexStr in trainingExamples) {
            let trainingExampleIndex = Number(trainingExampleIndexStr);

            let example = trainingExamples[trainingExampleIndex];
            let target = trainingTargets[trainingExampleIndex];

            let pred: number[] = this.predict(example);

            let outputErrorTerm = pred.map((pred, i) => (
                pred * (1 - pred) * (target[i] - pred)
            ));

            let errorTermsForLayers: [number[]] = [outputErrorTerm];

            for (let layerIndex = this.layers.length - 2; layerIndex > 0; layerIndex--) {
                errorTermsForLayers.unshift(this.layers[layerIndex].neurons.map((neuron, i) => {
                    let downstreamNeurons: Neuron[];

                    if (layerIndex == this.layers.length - 2) {
                        downstreamNeurons = this.layers[this.layers.length - 1].neurons;
                    } else {
                        downstreamNeurons = this.layers[layerIndex + 1].neurons;
                    }

                    let sumOfDownstreamWeightsAndErrorTerms: number = downstreamNeurons.map((neuron, j) => (
                        neuron.weights[j]
                    )).reduce((acc, curr, j) => {
                        if (j == 0) {
                            return acc * errorTermsForLayers[errorTermsForLayers.length - 1][j];
                        }

                        return acc + curr * errorTermsForLayers[errorTermsForLayers.length - 1][j];
                    });

                    return neuron.output * (1 - neuron.output) * sumOfDownstreamWeightsAndErrorTerms;
                }));
            }

            // Update weights
            for (let layerIndexStr in this.layers.slice(1)) {
                let layerIndex = Number(layerIndexStr);

                this.layers[layerIndex + 1].neurons.map((neuron, i) => {
                    neuron.weights = neuron.weights.map((neuronWeight, j) => (
                        neuronWeight + (this.learningRate * errorTermsForLayers[layerIndex][i] * neuron.inputs[j])
                    ));
                });
            }
        }
    }
}

let network: Network = new Network([2, 1], .05);

// Finish generating training data

let input = Array.apply(null, new Array(50000)).map(() => ([Math.round(Math.random()), Math.round(Math.random())]));
let output = input.map((i: number[]) => ([Number(i[0] == 1 || i[1] == 1)]));

network.train(input, output);
console.log(network.layers[network.layers.length - 1].neurons[0].weights)

console.log(network.predict([0, 0]));
console.log(network.predict([1, 0]));
console.log(network.predict([0, 1]));
console.log(network.predict([1, 1]));
