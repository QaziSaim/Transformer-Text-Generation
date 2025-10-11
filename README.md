# 🧠 Transformer Text Generation with TensorFlow & Keras

This project implements a **Transformer-based text generation model** from scratch using **TensorFlow** and **Keras**.
It trains on the **WikiText-2** dataset to learn next-word prediction and generate coherent English text sequences.

---

## 🚀 Project Overview

This project demonstrates:

* Building a **Transformer Encoder** model from scratch
* Understanding **multi-head attention**, **feed-forward layers**, and **embedding scaling**
* Training a model for **next-word prediction**
* Generating text autoregressively (like GPT-style models)

---

## 🧩 Architecture

The model follows the **original Transformer Encoder** structure:

* **Embedding Layer** – converts tokens to dense vectors
* **Positional Scaling** – scales embeddings by √d_model
* **Multi-Head Self-Attention** – learns contextual relationships
* **Feed Forward Network (FFN)** – adds non-linearity
* **Layer Normalization + Residual Connections**
* **Dense Output Layer** – predicts next token probabilities

---

## 📦 Dataset

**[WikiText-2](https://huggingface.co/datasets/wikitext)** from Hugging Face Datasets.

* Clean, high-quality English text from Wikipedia
* Used for language modeling and text generation

---

## 🧰 Requirements

Install dependencies:

```bash
pip install tensorflow datasets scikit-learn numpy
```

---

## 🧠 Training Pipeline

1. **Load Dataset**

   ```python
   from datasets import load_dataset
   dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
   ```

2. **Tokenize Text**

   ```python
   from tensorflow.keras.preprocessing.text import Tokenizer
   tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
   tokenizer.fit_on_texts(dataset['train']['text'])
   ```

3. **Create Sequences**

   * Input: previous 10 words
   * Output: next word

4. **Train-Test Split**

   ```python
   from sklearn.model_selection import train_test_split
   X_main, X_test, y_main, y_test = train_test_split(input_sequences, output_words, test_size=0.2, random_state=42)
   X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.25, random_state=42)
   ```

5. **Model Training**

   ```python
   model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=10,
       batch_size=64,
       callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
   )
   ```

6. **Evaluation**

   ```python
   loss, acc = model.evaluate(X_test, y_test)
   print(f"Test Accuracy: {acc:.4f}")
   ```

---

## ✍️ Text Generation

```python
def generate_text(seed_text, next_words=30):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0))
        output_word = next((w for w, i in tokenizer.word_index.items() if i == predicted), '')
        seed_text += " " + output_word
    return seed_text

print(generate_text("machine learning is"))
```

---

## 🧪 Results (Example)

After several epochs of training:

```
Epoch 5/10
Train Accuracy: ~0.20
Val Accuracy: ~0.18
```

Example output:

```
Input: "machine learning is"
Generated: "machine learning is a key method used in many artificial intelligence systems"
```

---

## ⚙️ Model Hyperparameters

| Parameter               | Value          |
| ----------------------- | -------------- |
| Embedding Dim (d_model) | 256            |
| Attention Heads         | 8              |
| Encoder Layers          | 3              |
| Feed-Forward Dim        | 1024           |
| Dropout Rate            | 0.1            |
| Sequence Length         | 10             |
| Optimizer               | Adam (lr=1e-4) |

---

## 📈 Future Improvements

* Add **Positional Encoding**
* Implement **Decoder** for full seq2seq generation
* Use **beam search** for better output diversity
* Experiment with **larger datasets** or **domain-specific data**

---

## 🧑‍💻 Author

**Sahim Qazi**
AI & Deep Learning Enthusiast | Machine Learning Engineer
💼 Focused on Transformer architectures, LLMs, and Generative AI.
