import fs from 'fs';
import { pipeline } from "@huggingface/transformers";

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
const SYSTEM_PROMPT = "You are an expert conversationalist who responds to the best of your ability. You can identify whether or not the input message from a social media platform contains a form of gender-based violence (GBV).";
// const MODEL_NAME = 'aggie/gbv-SmolLM2-135M-Instruct-ONNX';
const MODEL_NAME = 'aggie/gbv-Qwen2.5-0.5B-Instruct-ONNX';
const TSV_PATH = '/home/aj2066/browser-ml-inference/data/edos-test.tsv';
const BATCH_SIZE = 32;
const DTYPE = 'q4';

// Helper to safely stringify/shorten debug outputs
function short(obj, n = 300) {
  try {
    const s = typeof obj === 'string' ? obj : JSON.stringify(obj);
    return s.length > n ? s.slice(0, n) + '...' : s;
  } catch (e) {
    try { return String(obj); } catch (e2) { return '<unserializable>'; }
  }
}

function gbvPrompt(text) {
  return `Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output GBV if it contains GBV, or NotGBV if not. \nReturn ONLY one of: GBV, NotGBV. Do not add any other words.\n\n#### Input Text:\n${text}\n\n#### Answer:`;
}

function loadDataFromTSV(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n').filter(l => l.trim());
  if (lines.length === 0) throw new Error('TSV file is empty');
  const headers = lines[0].split('\t').map(h => h.trim());
  const textIdx = headers.indexOf('text');
  const labelIdx = headers.indexOf('label');
  if (textIdx === -1) throw new Error('No "text" column in TSV');
  const data = lines.slice(1).map(line => {
    const cols = line.split('\t');
    const text = (cols[textIdx] || '').trim();
    const rawLabel = labelIdx !== -1 ? (cols[labelIdx] || '').trim() : null;
    return { text, rawLabel };
  }).filter(d => d.text.length > 0);
  return data;
}

function mapLabel(label) {
  if (label === null || label === undefined || label === '') return null;
  const l = String(label).toLowerCase().trim();
  if (l === 'sexist' || l === '1' || l === 'gbv' || l === 'yes') return 'GBV';
  if (l === 'not sexist' || l === '0' || l === 'not gbv' || l === 'no') return 'NotGBV';
  return label;
}

// Batch inference + prediction postprocessing 
async function runGenerationInBatches(generator, inputTexts, labels) {
  const decoded_outputs = [];
  for (let i = 0; i < inputTexts.length; i += BATCH_SIZE) {
    const batch = inputTexts.slice(i, i + BATCH_SIZE);
    try {
      const res = await generator(batch, { max_new_tokens: 8, do_sample: false, temperature: 0.0 });
      for (let j = 0; j < res.length; j++) {
        const item = res[j];

        let raw_text = '';
        if (typeof item === 'string') raw_text = item;
        else if (Array.isArray(item)) {
          // Get assistant `content` inside `generated_text` message arrays
          let assistantContent = '';
          for (const el of item) {
            if (el && typeof el === 'object' && Array.isArray(el.generated_text)) {
              const msgs = el.generated_text;
              const assistantMsg = msgs.find(m => m && m.role === 'assistant') || msgs[msgs.length - 1];
              if (assistantMsg) {
                assistantContent = assistantMsg.content ?? assistantMsg.generated_text ?? assistantMsg.text ?? '';
                if (assistantContent) break;
              }
            }
          }
          if (assistantContent) {
            raw_text = assistantContent;
          } else {
            const parts = item.map(o => {
              if (typeof o === 'string') return o;
              if (!o || typeof o !== 'object') return String(o);
              return o.generated_text ?? o.text ?? o.content ?? o.output_text ?? JSON.stringify(o);
            });
            raw_text = parts.join('\n');
          }
        } else if (item && typeof item === 'object') raw_text = item.generated_text ?? item.text ?? item.content ?? item.output_text ?? JSON.stringify(item);
        else raw_text = String(item);
        console.log(`\nDEBUG raw_text [${i + j}]: type=${typeof raw_text} len=${raw_text.length} snippet=`, short(raw_text));
        
        const pieces = raw_text.split('Answer:');
        let suffix = pieces.length > 1 ? pieces.slice(-1)[0].trim() : raw_text.trim();
        // strip leading role markers like 'assistant'/'user'/'output'
        suffix = suffix.replace(/^\s*(assistant[:\s\n]+|assistant\s*|user[:\s\n]+|user\s*|output[:\s\n]+|output\s*)/i, '');
        let first_line = (suffix.split(/\r?\n/)[0] || '').trim();
        first_line = first_line.replace(/^[\W_]+/, '');

        const s_up = first_line.toUpperCase();
        console.log(`DEBUG s_up [${i + j}]:`, short(s_up));

        let pred_label = null;
        if (s_up.includes('NOTGBV')) {
          pred_label = 'NotGBV';
        } else if (s_up.includes('GBV')) {
          pred_label = 'GBV';
        } else if (/\bNOT\b|\bNO\b/.test(s_up)) {
          pred_label = 'NotGBV';
        } else if (/\bGBV\b|\bYES\b|\bY\b/.test(s_up)) {
          pred_label = 'GBV';
        } else {
          pred_label = 'NotGBV';
        }
        console.log(`DEBUG pred_label [${i + j}]:`, short(pred_label));

        const globalIdx = i + j;
        decoded_outputs.push({
          text: raw_text,
          prompt: raw_text.includes('Answer:') ? raw_text.split('Answer:')[0] : '',
          gbv: texts[globalIdx],
          output: first_line,
          prediction: pred_label,
          id: globalIdx,
          label: labels[globalIdx]
        });
      }
    } catch (err) {
      console.error('Generation batch error:', err);
      for (let k = 0; k < batch.length; k++) decoded_outputs.push({ text: '', prompt: '', gbv: texts[i + k], output: '', prediction: '', id: i + k, label: labels[i + k] });
    }
  }
  return decoded_outputs;
}

async function main() {
  const t0 = Date.now();
  console.log('Loading data...');
  const rows = loadDataFromTSV(TSV_PATH);
  const texts = rows.map(r => r.text);
  const labels = rows.map(r => mapLabel(r.rawLabel));
  console.log(`Loaded ${texts.length} rows`);

  const inputTexts = texts.map(t => [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: gbvPrompt(t) }
  ]);

  // // DEBUG: sample set of examples for debugging
  // const DEBUG_LIMIT = 10;
  // if (DEBUG_LIMIT && DEBUG_LIMIT > 0 && inputTexts.length > DEBUG_LIMIT) {
  //   console.log(`\nDEBUG mode: sampling first ${DEBUG_LIMIT} items for quicker runs`);
  //   inputTexts.splice(DEBUG_LIMIT); // keep first DEBUG_LIMIT
  //   texts.splice(DEBUG_LIMIT);
  //   labels.splice(DEBUG_LIMIT);
  // }

  console.log('\nCreating ONNX generation pipeline...');
  const generator = await pipeline(
    'text-generation', 
    MODEL_NAME, 
    { 
      backend: 'onnx', 
      dtype: DTYPE 
    }
  );

  console.log('Running generation in batches...');
  const decoded_outputs = await runGenerationInBatches(generator, inputTexts, labels);
  const predictions = decoded_outputs.map(d => d.prediction);
  // console.log(`\nDEBUG predictions: type=${typeof predictions} isArray=${Array.isArray(predictions)} length=${predictions.length}`);
  console.log('\nDEBUG predictions[0..10]:', short(predictions.slice(0, 10)));

  const validIndices = labels.map((l, i) => l !== null ? i : null).filter(i => i !== null);
  if (validIndices.length > 0) {
    let correct = 0;
    for (const idx of validIndices) {
      const pred = predictions[idx];
      const truth = labels[idx];
      if (pred === truth) correct += 1;
    }
    const acc = (correct / validIndices.length * 100).toFixed(2);
    console.log(`Accuracy: ${acc}% (${correct}/${validIndices.length})`);
  } else {
    console.log('No ground-truth labels present in TSV to compute accuracy');
  }

  console.log('\nExamples:');
  for (let i = 0; i < Math.min(5, texts.length); i++) {
    console.log(`- Text: ${texts[i].slice(0, 80)}...`);
    console.log(`  Label: ${labels[i]}`);
    console.log(`  Pred: ${predictions[i]}`);
  }

  const t1 = Date.now();
  console.log(`\nTotal prediction time: ${((t1 - t0) / 1000).toFixed(2)} seconds`);
}

main().catch(err => { console.error('Fatal error:', err); process.exit(1); });
