# Handwritten Drug Name Recognition (CRNN)

A deep learning project for recognizing handwritten drug names from word-level images using a CRNN (Convolutional Recurrent Neural Network).

---

## 🚀 Demo

### Input → Prediction

| Image | Prediction |
|------|-----------|
| ![](150.png) | bicozin |
| ![](265.png) | esoral |

---

## 🧠 What this does

- Takes a **single handwritten word image**
- Uses a **CRNN model (CNN + LSTM)**
- Outputs the predicted drug name

---

## ⚙️ How to run

```bash
python predict.py
```

## Make sure these files are in the same folder:

- crnn_prescription.pth
- predict.py
- sample image (e.g., 150.png)

---

## Project Structure

- crnn_prescription.pth   # trained model
- predict.py              # inference script
- medocr.ipynb            # training + fine-tuning
- 150.png                 # sample input
- 265.png                 # sample input
