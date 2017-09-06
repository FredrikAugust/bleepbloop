/*
Copyright 2017
Fredrik August Madsen-Malmo

bleepbloop.ts -- main file that binds the library together
*/

// Helper functions
function randomWeight(): number {
    return Math.random() * .2 - .1;
}


class Unit {
    weights: number;
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
            let layerIndex = Number(layerIndexStr);
            if (layerIndex == 0) { continue; }

            let layer = this.layers[layerIndex];

            layer.bias = randomWeight();

            for (let unit of layer.units) {
                unit.weights = Array.apply(null, new Array(this.layers[layerIndex - 1].units.length).map(() => (randomWeight())));
            }
        }
    }
}

let network: Network = new Network([1, 2, 3]);

for (let layer of network.layers) {
    console.log(layer.bias);
}
