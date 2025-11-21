
import { pipeline } from '@huggingface/transformers';

const INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output 1 if it contains GBV, or 0 if not.";
const CHOICES = "Choices: 1 for GBV, or 0 for Not GBV.";

function generatePrompt(text) {
  return `${INSTRUCTION} Text: ${text} ${CHOICES} Answer:`;
}

async function main() {
    const classifier = await pipeline(
        'text-classification', 
        'Heriot-WattUniversity/gbv-classifier-roberta-base-instruct-ONNX', 
        {
            backend: 'onnx', 
        //     revision: 'main',
        //     modelFileName: 'onnx/model_quantized.onnx',
            quantized: true,
            dtype: "q8", 
        }
    );
    
    const texts = [
        "Women should not be in politics.",
        "Everyone deserves respect and equality."
    ];
    console.log('Input texts:', texts);
    const promptTexts = texts.map(generatePrompt);
    console.log('Prompt texts:', promptTexts);

    try {
        const result = await classifier(promptTexts);
        console.log('Pipeline result:', result);
    } catch (err) {
        console.error('Pipeline error:', err);
        throw err;
    }
}

main();