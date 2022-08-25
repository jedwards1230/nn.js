/** Config for serialized Layer  */
type LayerConfig = {
	activation: string;
	inputs: number;
	outputs: number;
	lr: number;
	id: number;
	biases: null | number[];
	weights: null | number[][];
}

type LayerMapping = {
	[key: string]: any
}

/** Results of training session */
type TrainInfo = {
    time: number,
    loss: number,
    speed: number,
    distance: number,
    damaged: boolean,
    model: import("./network").Network,
}

type Generation = {
	id: number;
	distance: number;
	score: number;
}

type Loss = {
    loss: number,
    count: number,
}
