/*
Copyright 2017
Fredrik August Madsen-Malmo

bleepbloop.ts -- main file that binds the library together
*/

// Helper functions
function randomWeight(): number {
    return Math.random() * .2 - .1;
}

function sigmoid(input: number): number {
    return 1 / (1 + Math.pow(Math.E, input));
}


class Unit {
    weights: number[];

    calculate(inputs: number[], bias: number) {
        return sigmoid(inputs.reduce((acc, curr, i) => (
            acc + curr * this.weights[i]
        )) + bias);
    }
}

class Layer {
    units: Unit[];
    bias: number;

    constructor(unitCount: number) {
        this.units = Array.apply(null, new Array(unitCount)).map(() => (new Unit));
    }
}

class Network {
    layers: Layer[];

    constructor(public layerLenghts: number[]) {
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

            // doesn’t work
            for (let unit of layer.units) {
                unit.weights = Array.apply(null, new Array(this.layers[layerIndex - 1].units.length).map(() => (randomWeight())));
            }
            // end, broken stuff
        }
    }

    run(inputs: number[]) {
        if (inputs.length != this.layers[0].units.length) {
            console.error('Input length must be the same as number of input units!');
            return;
        }

        let prevLayerOutputs: number[];

        for (let layerIndexStr in this.layers) {
            let layerIndex: number = Number(layerIndexStr);

            if (layerIndex == 0) { prevLayerOutputs = inputs; }

            prevLayerOutputs = this.layers[layerIndex].units.map((unit: Unit) => (
                unit.calculate(prevLayerOutputs, this.layers[layerIndex].bias)
            ));

            console.log(prevLayerOutputs);
        }
    }
}

let network: Network = new Network([1, 2, 3]);

for (let layer of network.layers.slice(1)) {
    for (let unit of layer.units) {
        console.log(unit.weights);
    }
}

network.run([10]);
