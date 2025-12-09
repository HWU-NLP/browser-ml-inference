import { pipeline } from "@huggingface/transformers";

const SYSTEM = "You are an expert conversationalist who responds to the best of your ability. You can identify whether or not the input message from a social media platform contains a form of gender-based violence (GBV).";
const texts = [
        "Women should not be in politics.",  //yes
        // "Everyone deserves respect and equality.",  //no
        // "So stop being sexist towards women. Don't be that way. Be better.", //no
        // "Nothing more cucked than having your woman wear another man's jersey with his name on her back. Disgusting, foul bugmen!",  //yes
        // "Fucking skank. She should be in jail.", //yes
        // "Oh yes, she should be shot.", //no
        // "Absolute cuck move. Tell her to fuck off", //yes
        // "Cry about it incel bitches.", //no
    ];
const tsvFilePath = '/home/aj2066/browser-ml-inference/data/edos-test.tsv';
const modelName = 'aggie/gbv-SmolLM2-135M-Instruct-ONNX';
const batchSize = 32;
const quantized_version = "q4";

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
  return `${INSTRUCTION} Text: ${text} Answer:`;
}

function mapLabel(label) {
  if (label === null) return null;
  const lowerLabel = label.toLowerCase();
  if (lowerLabel === 'sexist' || lowerLabel === '1' || lowerLabel === 'gbv') {
    return 'GBV';
  } else if (lowerLabel === 'not sexist' || lowerLabel === '0' || lowerLabel === 'not gbv') {
    return 'NotGBV';
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
        'text-generation', 
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
        
    } catch (err) {
        console.error('Pipeline error:', err);
        throw err;
    }
    
    const endTime = performance.now();
    const elapsedSeconds = ((endTime - startTime) / 1000).toFixed(2);

    console.log(`Total execution time: ${elapsedSeconds} seconds`);
}

main();



// Create a text generation pipeline
const generator = await pipeline(
  "text-generation",
  "aggie/gbv-SmolLM2-135M-Instruct-ONNX",
  {
    backend: 'onnx', 
    dtype: "q4", 
    // device: 'webgpu',
  }
);

// Define the list of messages
const messages = [
  { role: "system", content: SYSTEM },
  { role: "user", content: "Women should not be in politics." },
];

// Generate a response
const output = await generator(messages, { max_new_tokens: 1 });
console.log(output[0].generated_text.at(-1).content);
// "The capital of France is Paris."
