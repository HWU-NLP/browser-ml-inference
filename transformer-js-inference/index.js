
import { pipeline } from '@xenova/transformers';

async function main() {
    // Load a pipeline using your custom ONNX model
    const classifier = await pipeline('text-classification', './models/gbv_classifier_int8.onnx', {
        // Optional: specify backend
        // backend: 'onnx', // default is ONNX
    });
    const result = await classifier('This is a test sentence.');
    console.log(result);
}

main();