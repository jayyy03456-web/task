# Customer Feedback Analyzer

A QLoRA fine-tune of Qwen2.5-7B-Instruct that reads an English product review and returns a single JSON object describing it. Given only a one-line instruction and nothing about formatting, the model outputs the sentiment, a 1 to 5 score, the topics mentioned, and a short summary, with no surrounding text.

The point of the project was to get dependable structured output from a short prompt. A general chat model will happily restate a review in prose, but a dashboard or database needs the same fields every time so a program can parse them. The output schema is fixed:

| Field | Meaning | Allowed values |
| --- | --- | --- |
| `sentiment` | overall sentiment | one of `positive`, `negative`, `neutral`, `mixed` |
| `score` | rating | integer from 1 to 5 |
| `topics` | aspects the review touches on | list of lowercase aspect words |
| `summary` | one-line summary | a single short English sentence |

Because the output is strict JSON, scoring is straightforward: parse it and check the fields, which gives a clean valid-JSON rate. On held-out reviews the base model produced valid JSON 0% of the time and answered every one in prose; after fine-tuning it was 100%.

The full write-up is in `report/Customer-Feedback-Analyzer-Finetuning-Report.docx`, and the training code is in `notebook/finetune.ipynb`.

## What fine-tuning changed

Under the same short prompt with no format instructions, the valid-JSON rate on held-out reviews went from 0% to 100%. The base model falls back on its chat habits and explains the review in a paragraph. After training, the habit of emitting the fixed structure sits in the weights, so the model produces parseable JSON from a one-line prompt that never mentions JSON. The four fields come back complete and legal, ready to feed downstream code.

## Tech stack

| Piece | Choice |
| --- | --- |
| Base model | Qwen2.5-7B-Instruct, loaded in 4-bit |
| Framework | Unsloth |
| Method | QLoRA: 4-bit frozen base plus LoRA adapters |
| Trainer | TRL `SFTTrainer` with `train_on_responses_only` |
| Chat template | qwen-2.5 (ChatML), matching the base model |
| Data | 280 synthetic review/JSON pairs, generated in code |
| Export | LoRA adapter and GGUF (`q4_k_m`) for Ollama or llama.cpp |

## Base model

The course suggested either an 8B-class instruct model or a smaller 0.5 to 3B one. We compared a few candidates and went with Qwen2.5-7B-Instruct.

| Model | Size | Note | Verdict |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | 7B | good English and Chinese; no thinking mode, so cleaner JSON; fits 16GB after 4-bit; mature tooling | chosen |
| Qwen2.5-3B-Instruct | 3B | lighter and faster, quality slightly lower | fallback |
| Qwen3-4B | 4B | newer, but its thinking mode can leak reasoning text into the JSON | not used |
| Llama-3.1-8B-Instruct | 8B | strong in English, weaker in Chinese, no edge here | not used |

The deciding factors were the lack of a thinking mode (no extra reasoning text bleeding into strict JSON), comfortable headroom in 16GB once quantized, and the largest pool of community fine-tuning material when something breaks.

## Repository layout

```
.
├── README.md
├── notebook/
│   └── finetune.ipynb          full pipeline, with outputs and the figure code
├── data/
│   └── dataset.jsonl           280 role-tagged training examples
├── img/                        figures written by the notebook
├── lora_model/                 exported LoRA adapter
├── model_gguf/                 exported GGUF (q4_k_m, for Ollama)
└── report/
    └── Customer-Feedback-Analyzer-Finetuning-Report.docx
```

## Setup

Everything ran on a single 16GB GPU (an RTX 5060 Ti, Blackwell architecture) on Windows. The versions that worked:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.12.0 (a build compiled for CUDA 12.x) |
| Triton | 3.7.0 |
| Unsloth | 2026.6.7 |
| Transformers | 5.5.0 |

Two things cost real time and are worth repeating. The card is Blackwell, so the default PyTorch was compiled only for older GPUs and could not see it; a build targeting a newer CUDA had to go in instead. Install order also matters. If you install Unsloth first and then run `pip install torch`, the second command overwrites the CUDA build with a CPU-only one and the GPU disappears. After each install step we confirmed that `torch.cuda.is_available()` returned `True` before continuing.

```bash
pip install unsloth
# if the GPU is not detected, install a PyTorch matching your CUDA first, then install unsloth last
```

The notebook also runs on a free Colab T4 if you would rather skip the local setup.

## Running the training

Open `notebook/finetune.ipynb` and run the cells top to bottom. The order is: load the 4-bit base, attach the LoRA adapter, load the data with the qwen-2.5 chat template, record the base model's answers, train, plot the loss, record the fine-tuned answers, compare the two, and export.

## Inference

Loading the adapter on top of the base model:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("lora_model", load_in_4bit=True)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a customer review analysis assistant."},
    {"role": "user", "content": "Battery dies in three hours and gets hot while charging."},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(inputs, max_new_tokens=160)[0], skip_special_tokens=True))
```

Running the exported GGUF with Ollama:

```bash
ollama create feedback-analyzer -f Modelfile
ollama run feedback-analyzer "The hotel location was great but the walls were paper thin."
```

## Dataset

There was no ready-made labeled corpus, so we generated 280 examples in code, each pairing a review with its target JSON. The generator holds twelve product categories (earbuds, laptops, running shoes, hotels, apps, coffee machines, and so on), a set of aspects per category, and positive and negative phrasings for every aspect. To build an example it picks a sentiment, selects a few aspects, writes a review that reads naturally, and derives the JSON label from the choices it just made. Mixed reviews are written as one positive sentence followed by a "However" turn so they do not look stitched together.

We cared more about variety across categories, sentiments, and phrasings than about raw count, and ran two checks before training. First, every review text is distinct, so repeated phrasings do not quietly gain weight. Second, every JSON parses, with all four fields present, a legal sentiment, an integer score, and a non-empty topics list. All 280 passed, which kept dirty data out of training.

For reference, the sentiment split is 94 positive, 87 negative, 66 mixed, and 33 neutral, and reviews run about 21 words on average (9 to 37).

## Hyperparameters

| Setting | Value | Reason |
| --- | --- | --- |
| LoRA rank `r` | 16 | safe default for a narrow task, enough capacity without bloat |
| `lora_alpha` | 16 | equal to the rank, a stable and common starting point |
| `lora_dropout` | 0 | Unsloth's recommended setting |
| `target_modules` | q, k, v, o, gate, up, down (7) | all attention and feed-forward projections |
| `learning_rate` | 2e-4 | a common rate for LoRA fine-tuning |
| `num_train_epochs` | 3 | enough to learn the format without over-memorizing |
| batch x grad-accum | 2 x 4 = 8 | fits 16GB while keeping training stable |
| `max_seq_length` | 2048 | well above review-plus-answer length, so no truncation |
| `optimizer` | adamw_8bit | 8-bit optimizer to save memory |
| scheduler | linear, 5 warmup steps | steady decay with a brief warmup |
| seed | 3407 | fixed for reproducibility |

Loss is computed only on the assistant's reply, so the system prompt and the review do not count toward it and the model spends every step learning to produce the JSON rather than echoing the input. Only the adapters train, about 40 million parameters, roughly half a percent of the base, while the rest stays frozen. The evaluation prompt is kept short and says nothing about format on purpose; that is what separates the two models, since the base has no reason to output JSON from it and the fine-tune does it anyway.

## Results

Tested on reviews that were not in the training data:

| Test set | Base | Fine-tuned |
| --- | --- | --- |
| Domain reviews (8) | 0% | 100% |
| General questions (4) | 0% | 25% |

The jump from 0 to 100 on domain reviews is the result we were after. The general questions are a forgetting check, and there JSON is not the goal, so the base model's 0 is the correct behavior. The fine-tuned model's 25 is format bleeding: it sometimes forces JSON onto unrelated questions, a known side effect worth flagging rather than hiding.

The same review through both models:

```
REVIEW       Battery dies in about three hours and it gets uncomfortably hot while charging.

BASE         It sounds like you're experiencing some significant issues with your device's
             battery life and heat generation during charging. Here's a breakdown...   (prose)

FINE-TUNED   {"sentiment": "negative", "score": 2, "topics": ["battery life",
             "heat during charging"], "summary": "Customer is dissatisfied, mainly due to
             battery life and heat."}                                                   (clean JSON)
```

## Figures

The notebook writes its plots to the `img/` folder. They are not embedded here; the files are:

| File | Shows |
| --- | --- |
| `img/01_dataset_composition.png` | sentiment counts and share |
| `img/02_review_length.png` | distribution of review lengths |
| `img/03_top_topics.png` | most frequent aspects |
| `img/04_training_loss.png` | training loss across steps |
| `img/05_json_rate_comparison.png` | valid-JSON rate, base vs fine-tuned |

The loss curve falls from about 3.35 to 0.01 over 105 steps, which is roughly three minutes of training. A final loss that low on a small set means the model has largely memorized the format, which is expected for this kind of task; if outputs start to feel rigid, drop an epoch or add more varied data.

## Limitations

The format bleeding above is the main one. The controlled vocabulary also costs some accuracy: because every review is mapped onto a fixed set of aspect words, unusual cases get flattened. The leaking blender, for example, came back as `neutral` with the topic `ease of use` and missed the actual complaint. The summaries read a little templated. And the dataset is small, so the near-zero loss reflects memorization more than proven generalization, which would need a larger and more realistic set to confirm.

A few things would help: mixing a small number of ordinary question-answer pairs into the training set so the model learns when not to use JSON; loosening the aspect vocabulary or pulling topics straight from the review; adding a held-out validation set with early stopping; and training the same data on a smaller model as a baseline.

## Export and local use

After training we save the LoRA adapter to `lora_model`. It is small, reloads on top of the base, and restores the fine-tuned behavior; this step ran reliably. We also export a GGUF (`q4_k_m`) to `model_gguf` for Ollama or llama.cpp. That step merges the adapter into the base, so it downloads the full-precision weights and converts locally, which needs a stable connection and some disk space. The export is wrapped so that if it fails the saved adapter is untouched and you can retry it later. Once the GGUF exists, a short Modelfile plus `ollama create` and `ollama run` is enough to load the model and chat with it locally.
