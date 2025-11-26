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
const modelName = 'Heriot-WattUniversity/gbv-classifier-roberta-base-instruct-ONNX';
// const modelName = 'Heriot-WattUniversity/gbv-classifier-Qwen2.5-0.5B-Instruct-ONNX';
const batchSize = 32;
const quantized_version = "q8";
// "modes": [
//         "fp16",
//         "q8",
//         "int8",
//         "uint8",
//         "q4",
//         "q4f16",
//         "bnb4"
//     ],

function loadDataFromTSV(filePath) {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const lines = fileContent.split('\n').filter(line => line.trim());
    
    if (lines.length === 0) {
      throw new Error('TSV file is empty');
    }
    
    const headers = lines[0].split('\t').map(h => h.trim());
    console.log('DEBUG: Headers found:', headers);
    console.log('DEBUG: Number of columns:', headers.length);
    
    const textColumnIndex = headers.indexOf('text');
    const labelColumnIndex = headers.indexOf('label');
    
    console.log('DEBUG: text column index:', textColumnIndex);
    console.log('DEBUG: label column index:', labelColumnIndex);
    
    if (textColumnIndex === -1) {
      throw new Error('No "text" column found in TSV file');
    }
    
    // Extract text and label values from all rows
    const data = lines.slice(1).map(line => {
      const columns = line.split('\t').map(c => c.trim());
      const text = columns[textColumnIndex] || '';
      const label = labelColumnIndex !== -1 ? columns[labelColumnIndex] : null;
      return { text, label };
    }).filter(item => item.text.trim());
    
    return data;
  } catch (err) {
    console.error(`Error loading TSV file: ${err.message}`);
    throw err;
  }
}

function generatePrompt(text) {
  return `${INSTRUCTION} Text: ${text} ${CHOICES} Answer:`;
}

function mapLabel(label) {
  if (label === null) return null;
  const lowerLabel = label.toLowerCase();
  if (lowerLabel === 'sexist' || lowerLabel === '1' || lowerLabel === 'gbv') {
    return 'GBV';
  } else if (lowerLabel === 'not sexist' || lowerLabel === '0' || lowerLabel === 'not gbv') {
    return 'Not GBV';
  }
  return label;
}

async function main() {
    const startTime = performance.now();
    console.log('Starting inference...\n');
    
    const data = loadDataFromTSV(tsvFilePath);
    const texts = data.map(d => d.text);
    const labels = data.map(d => mapLabel(d.label));
    
    console.log(`Loaded ${texts.length} texts from ${tsvFilePath}`);
    console.log('Sample input texts:', texts.slice(0, 3));
    console.log('Sample labels:', labels.slice(0, 3));

    const promptTexts = texts.map(generatePrompt);
    console.log('\nPrompt texts:', promptTexts.slice(0, 3));

    const classifier = await pipeline(
        'text-classification', 
        modelName,
        {
            backend: 'onnx', 
            dtype: quantized_version , 
            // device: 'webgpu',
        }
    );
    
    try {
        const result = await classifier(promptTexts, { batchSize: batchSize });
        console.log('\nPipeline result:', result.slice(0, 10)); // Log first 10 results
        
        // Calculate accuracy
        if (labels.some(l => l !== null)) {
            const predictions = result.map(r => r.label); 
            const groundTruth = labels;
            
            const validIndices = groundTruth.map((l, i) => l !== null ? i : null).filter(i => i !== null);
            const validPredictions = validIndices.map(i => predictions[i]);
            const validTruth = validIndices.map(i => groundTruth[i]);
            
            const correct = validPredictions.reduce((sum, pred, i) => sum + (pred === validTruth[i] ? 1 : 0), 0);
            const accuracy = (correct / validPredictions.length * 100).toFixed(2);
            
            console.log('\nDetailed predictions:');
            for (let i = 0; i < Math.min(10, texts.length); i++) {
                const pred = predictions[i];
                const truth = groundTruth[i];
                const match = pred === truth ? '✓' : '✗';
                console.log(`${match} Text: "${texts[i].substring(0, 50)}..." | Predicted: ${pred} | Truth: ${truth}`);
            }

            console.log(`\nModel used: ${modelName}`);
            console.log(`Quantized version used: ${quantized_version}`);
            console.log(`Accuracy: ${accuracy}% (${correct}/${validPredictions.length} correct)`);
        }
    } catch (err) {
        console.error('Pipeline error:', err);
        throw err;
    }
    
    const endTime = performance.now();
    const elapsedSeconds = ((endTime - startTime) / 1000).toFixed(2);

    console.log(`Total execution time: ${elapsedSeconds} seconds`);
}

main();
