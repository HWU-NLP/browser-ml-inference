import { pipeline } from '@huggingface/transformers';
import fs from 'fs';

const INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output 1 if it contains GBV, or 0 if not.";
const CHOICES = "Choices: 1 for GBV, or 0 for Not GBV.";

// const texts = [
//         "Women should not be in politics.",  //yes
//         "Everyone deserves respect and equality.",  //no
//         "So stop being sexist towards women. Don't be that way. Be better.", //no
//         "Nothing more cucked than having your woman wear another man's jersey with his name on her back. Disgusting, foul bugmen!",  //yes
//         "Fucking skank. She should be in jail.", //yes
//         "Oh yes, she should be shot.", //no
//         "Absolute cuck move. Tell her to fuck off", //yes
//         "Cry about it incel bitches.", //no
//     ];

const tsvFilePath = '/home/aj2066/browser-ml-inference/data/edos-test.tsv';
const batchSize = 32;

function loadDataFromTSV(filePath) {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const lines = fileContent.split('\n').filter(line => line.trim());
    
    if (lines.length === 0) {
      throw new Error('TSV file is empty');
    }
    
    // Parse header to find 'text' column index
    const headers = lines[0].split('\t');
    const textColumnIndex = headers.indexOf('text');
    
    if (textColumnIndex === -1) {
      throw new Error('No "text" column found in TSV file');
    }
    
    // Extract text values from all rows
    const texts = lines.slice(1).map(line => {
      const columns = line.split('\t');
      return columns[textColumnIndex] || '';
    }).filter(text => text.trim());
    
    return texts;
  } catch (err) {
    console.error(`Error loading TSV file: ${err.message}`);
    throw err;
  }
}

function generatePrompt(text) {
  return `${INSTRUCTION}sText: ${text} ${CHOICES} Answer:`;
}

async function main() {
    const startTime = performance.now();
    console.log('Starting inference...\n');
    
    const texts = loadDataFromTSV(tsvFilePath);
    console.log(`Loaded ${texts.length} texts from ${tsvFilePath}`);
    console.log('Sample input texts:', texts.slice(0, 5));

    const classifier = await pipeline(
        'text-classification', 
        'Heriot-WattUniversity/gbv-classifier-roberta-base-instruct-ONNX', 
        // 'Heriot-WattUniversity/gbv-classifier-Qwen2.5-0.5B-Instruct-ONNX',
        {
            backend: 'onnx', 
            dtype: "q8", 
            // device: 'webgpu',
        }
    );
    
    const promptTexts = texts.map(generatePrompt);
    console.log('Prompt texts:', promptTexts.slice(0, 5));

    try {
        const result = await classifier(promptTexts, { batchSize: batchSize });
        console.log('Pipeline result:', result.slice(0, 20)); // Log first 20 results
    } catch (err) {
        console.error('Pipeline error:', err);
        throw err;
    }
    
    const endTime = performance.now();
    const elapsedSeconds = ((endTime - startTime) / 1000).toFixed(2);
    console.log(`\nTotal execution time: ${elapsedSeconds} seconds`);
}

main();
