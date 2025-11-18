
import { pipeline } from '@huggingface/transformers';

async function main() {
    const classifier = await pipeline(
        'text-classification', 
        'Heriot-WattUniversity/gbv-classifier-roberta-base-ONNX', 
        {
            backend: 'onnx', 
        //     revision: 'main',
        //     modelFileName: 'onnx/model_quantized.onnx',
            quantized: true,
            dtype: "q8", 
        }
    );
    
    try {
        const result = await classifier(['I hate you!', 'here is a neutral statement.']);
        console.log('Pipeline result:', result);
    } catch (err) {
        console.error('Pipeline error:', err);
        throw err;
    }
}

main();