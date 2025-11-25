import { pipeline } from '@huggingface/transformers';

const INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output 1 if it contains GBV, or 0 if not.";
const CHOICES = "Choices: 1 for GBV, or 0 for Not GBV.";

const texts = [
        "Women should not be in politics.",  //yes
        "Everyone deserves respect and equality.",  //no
        "So stop being sexist towards women. Don't be that way. Be better.", //no
        "Nothing more cucked than having your woman wear another man's jersey with his name on her back. Disgusting, foul bugmen!",  //yes
        "Fucking skank. She should be in jail.", //yes
        "Oh yes, she should be shot.", //no
        "Absolute cuck move. Tell her to fuck off", //yes
        "Cry about it incel bitches.", //no
    ];
const batchSize = 4;

function generatePrompt(text) {
  return `${INSTRUCTION} Text: ${text} ${CHOICES} Answer:`;
}

async function main() {
    const classifier = await pipeline(
        'text-classification', 
        'Heriot-WattUniversity/gbv-classifier-roberta-base-instruct-ONNX', 
        // 'Heriot-WattUniversity/gbv-classifier-Qwen2.5-0.5B-Instruct-ONNX',
        {
            backend: 'onnx', 
        //     revision: 'main',
        //     modelFileName: 'onnx/model_quantized.onnx',
            quantized: true,
            dtype: "q8", 
        }
    );
    
    console.log('Input texts:', texts);
    const promptTexts = texts.map(generatePrompt);
    console.log('Prompt texts:', promptTexts);

    try {
        const result = await classifier(promptTexts, { batchSize: batchSize });
        console.log('Pipeline result:', result);
    } catch (err) {
        console.error('Pipeline error:', err);
        throw err;
    }
}

main();