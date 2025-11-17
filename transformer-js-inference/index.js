
import { pipeline } from '@huggingface/transformers';

async function main() {
    // Load a pipeline using your custom ONNX model
    const classifier = await pipeline(
        'text-classification', 
        'Heriot-WattUniversity/gbv-classifier-roberta-base', 
        {
            backend: 'onnx', 
            revision: 'main',
            modelFileName: 'onnx/model.onnx',
            // modelFileName: 'onnx/model_quantized.onnx',
            // quantized: true,
            // dtype: "q8", 
        }
    );
    const result = await classifier('I hate you!');
    console.log(result);
}

main();