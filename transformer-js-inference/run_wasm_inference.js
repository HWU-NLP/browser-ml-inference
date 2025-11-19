import { AutoTokenizer } from '@huggingface/transformers';
import { env } from '@xenova/transformers';
import * as ortWeb from 'onnxruntime-web';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Optional: point to local wasm files shipped with @xenova/transformers
try {
  env.backends.onnx.wasm.wasmPaths = __dirname + '/node_modules/@xenova/transformers/dist/';
} catch (e) {
  // ignore if env not writable
}

const INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output A if it contains GBV, or B if not.";
const CHOICES = "Choices: 1 for GBV, or 0 for Not GBV.";

function generatePrompt(text) {
  return `${INSTRUCTION} Text: ${text} ${CHOICES} Answer:`;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a,b) => a + b, 0);
  return exps.map(e => e / sum);
}

async function main() {
  const texts = [
    "Women should not be in politics.",
    "Everyone deserves respect and equality."
  ];
  const modelName = 'Heriot-WattUniversity/gbv-classifier-roberta-base-onnx';
  const tokenizer = await AutoTokenizer.from_pretrained(modelName);

  const promptTexts = texts.map(generatePrompt);
  console.log('Prompt texts:', promptTexts);

  const tokenized = await tokenizer(promptTexts, {
    padding: 'max_length',
    truncation: true,
    max_length: 512,
  });
  console.log('Tokenized inputs:', tokenized);

  // Prefer loading the model directly from the Hugging Face repo for `modelName`.
  // Try quantized first, then non-quantized. Fall back to local repo ONNX if remote not available.
  const hfBase = `https://huggingface.co/${modelName}/resolve/main/onnx/`;
  const remoteQuant = hfBase + 'model_quantized.onnx';
  const remoteModel = hfBase + 'model.onnx';

  async function remoteExists(url) {
    try {
      const res = await fetch(url, { method: 'HEAD' });
      return res.ok;
    } catch (e) {
      return false;
    }
  }

  let modelSource;
  if (await remoteExists(remoteQuant)) {
    modelSource = remoteQuant;
  } else if (await remoteExists(remoteModel)) {
    modelSource = remoteModel;
  } else {
    throw new Error(`No remote ONNX model found for ${modelName}.`);   
  }

  console.log('Using model ->', modelSource);

  // If modelSource is a remote URL, fetch into memory and pass buffer to InferenceSession.create
  let modelLoadSource = modelSource;
  if (typeof modelSource === 'string' && (modelSource.startsWith('http://') || modelSource.startsWith('https://'))) {
    const resp = await fetch(modelSource);
    if (!resp.ok) throw new Error(`Failed to download remote model: ${resp.status} ${resp.statusText}`);
    const ab = await resp.arrayBuffer();
    modelLoadSource = new Uint8Array(ab);
  }

  // Create WASM session from the model buffer or local path
  const session = await ortWeb.InferenceSession.create(modelLoadSource, { executionProviders: ['wasm'] });

  const inputIdsFlat = tokenized.input_ids.ort_tensor.cpuData;
  const attentionFlat = tokenized.attention_mask.ort_tensor.cpuData;

  const inputIdsBig = BigInt64Array.from(Array.from(inputIdsFlat).map(x => BigInt(x)));
  const attentionBig = BigInt64Array.from(Array.from(attentionFlat).map(x => BigInt(x)));

  const batchSize = promptTexts.length;
  const seqLen = inputIdsBig.length / batchSize;

  const feeds = {
    input_ids: new ortWeb.Tensor('int64', inputIdsBig, [batchSize, seqLen]),
    attention_mask: new ortWeb.Tensor('int64', attentionBig, [batchSize, seqLen]),
  };

  const out = await session.run(feeds);
  const logits = out[Object.keys(out)[0]].data;

  // logits is a flat array: [logit0_label0, logit0_label1, logit1_label0, logit1_label1, ...]
  const numLabels = 2;
  const predictions = [];
  for (let i = 0; i < logits.length; i += numLabels) {
    const slice = Array.from(logits).slice(i, i + numLabels);
    const probs = softmax(slice);
    const pred = probs.indexOf(Math.max(...probs));
    predictions.push({ probs, pred });
  }

  console.log('Predictions:');
  predictions.forEach((p, i) => {
    console.log(`- ${texts[i]} -> label=${p.pred} (probabilities=${p.probs.map(v => v.toFixed(4))})`);
  });
}

main().catch(err => { console.error(err); process.exit(1); });
