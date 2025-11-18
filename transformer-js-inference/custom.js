import { AutoTokenizer, pipeline } from '@huggingface/transformers';
import { env } from '@xenova/transformers';
// Import ONNX backend methods
// const { createInferenceSession, runInferenceSession } = env.backends.onnx;

console.log(env.backends);

const INSTRUCTION = "Classify the following message from a social media platform. It might contain a form of gender-based violence (GBV). Output A if it contains GBV, or B if not.";
const CHOICES = "Choices: 1 for GBV, or 0 for Not GBV.";

function generatePrompt(text) {
  return `${INSTRUCTION} Text: ${text} ${CHOICES} Answer:`;
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
  const result = await classifier(promptTexts);
  console.log(result);

  // // create ONNX inference session
  // console.log(env.backends.onnx.createInferenceSession)
  // const session = await env.backends.onnx.createInferenceSession(modelName, {
  //   modelFileName: 'onnx/model.onnx',
  //   // modelFileName: 'onnx/model_quantized.onnx',
  //   // quantized: true,
  //   // dtype: "q8",
  // });

  // const input_ids = new BigInt64Array(tokenized.input_ids.flat());
  // const attention_mask = new BigInt64Array(tokenized.attention_mask.flat());

  // const ortFeed = {
  //   input_ids: new ort.Tensor('int64', input_ids, [texts.length, 512]),
  //   attention_mask: new ort.Tensor('int64', attention_mask, [texts.length, 512]),
  // };
  // console.log('ONNX Runtime feed:', ortFeed);

//   const outputs = await env.backends.onnx.runInferenceSession(session, ortFeed);

//   const logits = Object.values(outputs)[0].data;

//   // Convert to per-row softmax
//   const numLabels = 2;
//   const predictions = [];
//   for (let i = 0; i < logits.length; i += numLabels) {
//     const slice = logits.slice(i, i + numLabels);
//     const exp = slice.map(Math.exp);
//     const sum = exp.reduce((a, b) => a + b, 0);
//     const probs = exp.map(v => v / sum);
//     const pred = probs.indexOf(Math.max(...probs));
//     predictions.push(pred);
//   }

//   console.log('Predictions:', predictions);
//   const labels = [1, 0];
//   const accuracy =
//     predictions.filter((p, i) => p === labels[i]).length / labels.length;
//   console.log('Accuracy:', accuracy);
}

main();
