import { AppConfig } from "./config";
import {
    LayerMap,
    Layer,
} from "./layer";

export function MSE(targets: number, outputs: number) {
    return (targets - outputs) ** 2;
}

export class Network {
    name: string;
    config: AppConfig;
    epsilon: number;
    lr: number;
    alias: string;
    memory: any[];
    layers: Layer[];
    confidence: number;
    lossFunction: (targets: number[], outputs: number[]) => number;
    deriveLoss: (targets: number[], outputs: number[]) => number[];

    constructor(modelConfig: AppConfig) {
        this.memory = [];
        this.confidence = 0.5;
        this.config = modelConfig;
        this.name = modelConfig.name;
        this.lr = modelConfig.lr;
        this.epsilon = modelConfig.epsilonDecay;
        this.alias = modelConfig.alias;

        this.layers = new Array(modelConfig.layers.length);
        for (let i = 0; i < modelConfig.layers.length; i++) {
            const layerConfig = modelConfig.layers[i];
            layerConfig.lr = this.lr;
            this.layers[i] = new LayerMap[layerConfig.activation](layerConfig);
        }

        this.lossFunction = (targets, outputs) => {
            let cost = 0
            for (let i = 0; i < outputs.length; i++) {
                cost += MSE(targets[i], outputs[i]);
            }
            return cost / outputs.length;
        };

        this.deriveLoss = (targets, outputs) => {
            const derivatives = [];
            for (let i = 0; i < outputs.length; i++) {
                derivatives[i] = (targets[i] - outputs[i]) * 2;
            }
            return derivatives;
        }
    }

    /** Forward pass each layer */
    public forward(inputs: number[], backprop = false): number[] {
        let outputs = this.layers[0].forward(inputs, backprop);
        for (let i = 1; i < this.layers.length; i++) {
            outputs = this.layers[i].forward(outputs, backprop);
        }
        return outputs;
    }

    /** Backward pass each layer */
    public backward(delta: number[]) {
        for (let i = this.layers.length - 1; i >= 0; i--) {
            delta = this.layers[i].backward(delta);
        }
    }

    /** Choose action based on confidence or with epsilon greedy */
    public makeChoice(outputValues: number[], greedy = false) {
        // choose random
        const random = Math.random();
        if (greedy && (random < this.epsilon)) {
            for (let i = 0; i < outputValues.length; i++) {
                outputValues[i] = (Math.random() * outputValues.length) | 0;
            }
        }
        this.decay();
        return outputValues;
    }

    public saveLayers() {
        const layers: LayerConfig[] = new Array(this.layers.length);
        for (let i = 0; i < this.layers.length; i++) {
            const level = this.layers[i];
            level.id = i;
            layers.push(level.save());
        }

        return layers
    }

    /** Slightly mutate weights for model */
    public mutate(amount: number = 0.1, rate: number = 0): Network {
        const mutated = new Network(this.config);
        for (let i = 0; i < mutated.layers.length; i++) {
            mutated.layers[i].mutate(amount, rate);
        }
        return mutated;
    }

    private decay() {
        // epsilon decay
        if (this.epsilon > 0.01) this.epsilon *= 0.99;

        // learning rate decay
        for (let i = this.layers.length - 1; i >= 0; i--) {
            this.layers[i].lr = this.layers[i].lr > 0.0001 ? this.layers[i].lr * 0.99 : 0.00001;
        }
    }
}
