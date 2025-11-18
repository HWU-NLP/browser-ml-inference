
import { pipeline } from '@huggingface/transformers';

async function main() {
    const classifier = await pipeline(
        'text-classification', 
        'nicky48/toxic-bert-ONNX', //'Heriot-WattUniversity/gbv-classifier-roberta-base-onnx', 
        {
            backend: 'onnx', 
            // modelFileName: 'onnx/model_quantized.onnx',
            quantized: true,
            dtype: "q8", 
        }
    );
    const result = await classifier('I hate you!');
    const { label, score } = result[0];

    console.log(label);
    console.log(score);
}

main();