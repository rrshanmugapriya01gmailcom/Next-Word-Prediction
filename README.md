# 🔮 Next Word Prediction using LSTM & GPT-2 Transformers (Fine-Tuned on WikiText-2)

This project explores and compares two deep learning approaches for **Next Word Prediction**:

- 📚 **LSTM-based model** (built from scratch using Keras)
- 🤖 **Transformer-based model** (fine-tuned GPT-2 using Hugging Face Transformers)

Both models are trained on the [WikiText-2](https://huggingface.co/datasets/wikitext) dataset and evaluated on **loss** and **perplexity** to determine performance.

---

## 🚀 Project Structure

```
📁 next_word_prediction_project/
│
├── lstm_model/
│   ├── final_lstm_model.h5
│   ├── model.weights.h5
│   └── tokenizer.pkl
│
├── transformers_model/
│   └── gpt2-wikitext-best-model/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer.json
│
├── comparison_plot.png
├── README.md
├── requirements.txt
```

---

## 🧠 Models Used

### ✅ LSTM (Keras)
- Tokenized using `Tokenizer`
- Created n-gram sequences of size 4
- Trained with a Bidirectional LSTM layer
- Metrics: `accuracy`, `top_5_accuracy`
- Saved: `.h5` model, tokenizer, and weights

### ✅ GPT-2 Transformer (Hugging Face)
- Fine-tuned using `Trainer` on grouped token blocks
- Used `GPT2Config` with dropout adjustment
- Early stopping, cosine learning rate scheduling
- Evaluation Metric: `eval_loss` and Perplexity
- Saved best model and tokenizer

---

## 📊 Model Comparison

| Metric       | LSTM Model | GPT-2 Transformer |
|--------------|------------|-------------------|
| **Loss**     | 4.5        | 3.065             |
| **Perplexity** | ~90       | **21.44**         |

> GPT-2 outperforms LSTM on loss and perplexity.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/next-word-prediction.git
cd next-word-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

Dataset used: **WikiText-2**

```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```

---

## 🏋️‍♂️ Training

### ➤ LSTM Training

```bash
python train_lstm_model.py
```

Saves to:
- `final_lstm_model.h5`
- `tokenizer.pkl`
- `model.weights.h5`

### ➤ GPT-2 Fine-Tuning

```bash
python train_transformer.py
```

Outputs:
- `transformers_model/gpt2-wikitext-best-model/`

---

## 🧪 Inference

### ➤ Using Fine-Tuned GPT-2

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("transformers_model/gpt2-wikitext-best-model")
tokenizer = AutoTokenizer.from_pretrained("transformers_model/gpt2-wikitext-best-model")

prompt = "The history of India begins with"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

---

## 📈 Results Visualization

![Comparison](comparison_plot.png)

---

## 📌 Key Learnings

- LSTM is simple and requires manual preprocessing.
- Transformers handle longer context and outperform in prediction accuracy.
- Hugging Face tools simplify training and evaluation pipelines.

---

## 🧠 Authors & Credits

- **Developer:** R R Shanmugapriya
- **Dataset:** [WikiText-2](https://huggingface.co/datasets/wikitext)
- **Frameworks:** TensorFlow, PyTorch, Hugging Face Transformers

---

## 📄 License

This project is licensed under the MIT License.
