/*
Copyright 2017
Fredrik August Madsen-Malmo

bleepbloop.ts -- main file that binds the library together
*/

/*

Reduce doesn’t work if the array that is being iteratied over only contains
one element.

*/

// Helper functions
function randomWeight(): number {
    return Math.random() * .2 - .1;
}

function sigmoid(input: number): number {
    return 1 / (1 + Math.pow(Math.E, -1 * input));
}

function sumOfSquaredError(target: number[], pred: number[]): number {
    // Issue #5
    return target.reduce((acc, curr, i) => {
        if (i == 1) {
            return Math.pow(acc - pred[i - 1], 2) + Math.pow(curr - pred[i], 2);
        }

        return acc + Math.pow(curr - pred[i], 2);
    })
}

class Neuron {
    weights: number[];
    output: number;
    inputs: number[];

    activate(inputs: number[], bias: number): number {
        // This makes it easier to retrieve the input the neuron received when we’re backpropagating
        this.inputs = inputs;

        // Calculate the net input (Sigma[inputs]w*x)+b
        // Issue #5
        return inputs.reduce((acc, curr, i) => {
            if (i == 1) {
                return acc * this.weights[i - 1] + curr * this.weights[i];
            }

            return acc + curr * this.weights[i];
        }) + bias;
    }

    transfer(inputs: number[], bias: number): number {
        // Save the output of the neuron, for backpropagation as well
        return (this.output = sigmoid(this.activate(inputs, bias)));
    }
}

class Layer {
    neurons: Neuron[];
    bias: number;

    constructor(neuronCount: number) {
        // Instanciate neuronCount neurons and save them in this.neurons
        this.neurons = Array.apply(null, new Array(neuronCount)).map(() => (new Neuron));
    }
}

class Network {
    layers: Layer[];

    constructor(layerLenghts: number[], public learningRate: number) {
        // Instanciate layerLenghts.length layers and save them in this.neurons
        this.layers = layerLenghts.map((layerLength, i) => {
            return new Layer(layerLenghts[i]);
        });

        this.seedWeightsAndBias();
    }

    seedWeightsAndBias(): void {
        for (let layerIndexStr in this.layers) {
            let layerIndex: number = Number(layerIndexStr);

            // Since the input layer doesn’t have any weights/bias
            if (layerIndex == 0) { continue; }

            let layer: Layer = this.layers[layerIndex];

            layer.bias = randomWeight();

            // Assign random weights to each of the neurons
            for (let neuron of layer.neurons) {
                neuron.weights = Array.apply(null, new Array(this.layers[layerIndex - 1].neurons.length)).map(() => (
                    randomWeight()
                ));
            }
        }
    }

    predict(inputs: number[]): number[] {
        if (inputs.length != this.layers[0].neurons.length) {
            console.error('Input length must be the same as number of input neurons!');
            return [];
        }

        // The output from the layer before the one we’re working with
        let prevLayerOutputs: number[] = inputs;

        for (let layerIndexStr in this.layers) {
            let layerIndex: number = Number(layerIndexStr);

            // Skip input layer, as it’s only function is to pass on the input
            if (layerIndex == 0) {
                continue;
            }

            // Activate all of the neurons with the previous output(prevLayerOutputs) and pass
            // through sigmoid function.
            prevLayerOutputs = this.layers[layerIndex].neurons.map((neuron) => (
                neuron.transfer(prevLayerOutputs, this.layers[layerIndex].bias)
            ));
        }

        return prevLayerOutputs;
    }

    train(trainingExamples: [number[]], trainingTargets: [number[]], iterations: number, errorStopValue: number, minimumEpochs: number = 0) {
        // Using backpropagation
        if (trainingExamples[0].length != this.layers[0].neurons.length) {
            console.error('Input length must be the same as number of input neurons!');
            return;
        }

        if (trainingTargets[0].length != this.layers[this.layers.length - 1].neurons.length) {
            console.error('Target length must be the same as number of utput neurons!');
            return;
        }

        if (trainingExamples.length != trainingTargets.length) {
            console.error('Examples must be of same length as targets!');
            return;
        }

        for (let epoch = 0; epoch < iterations; epoch++) {
            let trainingIndex: number = Math.floor(Math.random() * trainingExamples.length);

            // Extract an example(inputs) and a target
            let example = trainingExamples[trainingIndex];
            let target = trainingTargets[trainingIndex];

            let pred: number[] = this.predict(example);


            let outputErrorTerm = pred.map((pred, i) => (
                pred * (1 - pred) * (target[i] - pred)
            ));

            // Instanciate errorTermsForLayers with the output error terms, as this has it’s
            // own function
            let errorTermsForLayers: [number[]] = [outputErrorTerm];

            // Loop backwards through the layers excluding output and input layers
            for (let layerIndex = this.layers.length - 2; layerIndex > 0; layerIndex--) {
                // Prepend the error terms for the layer to the errorTermsForLayers
                // since we’re looping backwards
                errorTermsForLayers.unshift(this.layers[layerIndex].neurons.map((neuron, i) => {
                    // These are the neurons who take the output from this neuron as input (directly)
                    let downstreamNeurons: Neuron[] = this.layers[layerIndex + 1].neurons;

                    let sumOfDownstreamWeightsAndErrorTerms: number =
                        // First, isolate the weights as we can’t reduce Neuron[] with number as acc
                        downstreamNeurons.map((neuron, j) => (
                            neuron.weights[j]
                            // Issue #5
                        )).reduce((acc, curr, j) => {
                            if (j == 1) {
                                return acc * errorTermsForLayers[errorTermsForLayers.length - 1][j - 1] +
                                    curr * errorTermsForLayers[errorTermsForLayers.length - 1][j];
                            }

                            return acc + curr * errorTermsForLayers[errorTermsForLayers.length - 1][j];
                        });

                    // Derivation of the sigmoid function
                    return neuron.output * (1 - neuron.output) * sumOfDownstreamWeightsAndErrorTerms;
                }));
            }

            // Update weights
            // Skip the input layer, hence the +1 in the body
            for (let layerIndexStr in this.layers.slice(1)) {
                let layerIndex: number = Number(layerIndexStr);

                // Update the weights of the neurons in this layer
                this.layers[layerIndex + 1].neurons.map((neuron, i) => {
                    neuron.weights = neuron.weights.map((neuronWeight, j) => {
                        // Update bias for layer. This is the same as weight update rule, except the
                        // input weight is 1 in the calculation, as bias is directly passed in
                        this.layers[layerIndex + 1].bias +=
                            this.learningRate * 1 * errorTermsForLayers[layerIndex][i];

                        // Update neuron weight based on the error term for the layer times
                        // the input value
                        return neuronWeight +
                            (this.learningRate * errorTermsForLayers[layerIndex][i]
                                * neuron.inputs[j])
                    });
                });
            }

            // epochSumOfSquaredError
            let epochSSE = sumOfSquaredError(target, pred)
            if (epochSSE <= errorStopValue && epoch >= minimumEpochs) {
                console.info("Neural network done training, epoch: " + epoch + "\nSSE=" + epochSSE);
                break;
            }
        }
    }
}

/* Testing the program */

let network: Network = new Network([2, 1], .25);

// Finish generating training data

let input: [number[]] = [[0, 0], [0, 1], [1, 0], [1, 1]];
let output: [number[]] = [[0], [1], [1], [1]];

network.train(input, output, 10000, 0.01, 500);

console.log(network.predict([0, 0]));
console.log(network.predict([1, 0]));
console.log(network.predict([0, 1]));
console.log(network.predict([1, 1]));
