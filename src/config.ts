export class AppConfig {
	name: string;
	alias: string;
	lr: number;
	layers: LayerConfig[];
	generations: Generation[];
	epsilonDecay: number;
	mutationAmount: number;
	mutationRate: number;
	sensorCount: number;

	constructor(name: string, alias: string) {
		this.name = name;
		this.alias = alias;
		this.lr = 0.01;
		this.generations = [];
		this.epsilonDecay = 0.99;
		this.mutationAmount = 0.2;
		this.mutationRate = 0.005;
		this.sensorCount = 7;
		this.layers = [
			{
				id: 0,
				activation: "Tanh",
				inputs: 10,
				outputs: 15,
				lr: 0.01,
				biases: null,
				weights: null,
			},
			{
				id: 1,
				activation: "Tanh",
				inputs: 15,
				outputs: 10,
				lr: 0.01,
				biases: null,
				weights: null,
			},
			{
				id: 2,
				activation: "Sigmoid",
				inputs: 10,
				outputs: 4,
				lr: 0.01,
				biases: null,
				weights: null,
			},
		];
	}
}