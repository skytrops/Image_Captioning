# 🖼️ Image Captioning using Python  
Using **Inception V3** as Encoder and **LSTM** as Decoder  

---

## 📌 Problem Statement  
The process of creating a written description of an image is known as **image captioning**.  
Dataset format: **[picture → caption]**.  

The dataset consists of input photographs and their corresponding captions.  

---

## ⚙️ How does it work?  

### 🧠 CNN (Encoder)  
A Convolutional Neural Network (CNN) takes an input image, extracts important features, and produces a feature vector.  
- Here, **Inception V3** is used (last softmax removed).  
- Output is a feature vector of size **1×1×2048**.  
- This vector is mapped via a linear layer to match LSTM input size.  

### ✍️ LSTM (Decoder)  
- RNNs suffer from long-term dependency issues → solved using **LSTMs**.  
- LSTMs can handle sequential data (text, speech, time series).  
- In image captioning, the LSTM is trained as a **language model conditioned on the image features**.  

**Example**:  
If the caption is *“Giraffes standing next to each other”*  
- Source sequence: `<start>, Giraffes, standing, next, to, each, other`  
- Target sequence: `Giraffes, standing, next, to, each, other, <end>`  

---

## 🏗️ Model Architecture  
- **Encoder**: Pre-trained **Inception V3** extracts image features.  
- **Decoder**: LSTM generates captions word by word.  
- Training is done using predefined `<start>` and `<end>` tokens.  

---

## 📂 Dataset  
We use the **Flickr8k Dataset**, freely available on Kaggle:  
🔗 [Flickr8k Dataset](https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb)  

**Details:**  
- 8092 images (JPEG format, varied sizes).  
- Each image has **5 captions** → total **40,460 captions**.  
- Dataset split: Training / Testing / Development.  
- Text annotations are stored in **Flickr8k.token.txt**.  

---

## 🏋️ Training  
- Model trained for **80 epochs**.  
- Training includes backpropagation through both CNN & LSTM.  
- Evaluation metric: **BLEU Score**.  

### 📊 BLEU Score  
- Measures similarity between generated and reference captions.  
- Range: **0.0 → 1.0** (1.0 = perfect match).  

---

## 🚀 Implementation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/skytrops/Image_Captioning.git
   cd Image_Captioning
