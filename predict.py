
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# -------------------------
# VOCAB
# -------------------------
import string
vocab = list(string.ascii_lowercase)
vocab.append("<blank>")

char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}

# -------------------------
# MODEL
# -------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            input_size=256 * 8,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.view(b, w, c * h)

        x, _ = self.rnn(x)
        x = self.fc(x)

        return x

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((32, 256)),
    transforms.ToTensor()
])

# -------------------------
# DECODE
# -------------------------
def decode_predictions(outputs, idx2char, blank):
    preds = outputs.argmax(2)
    preds = preds.permute(1, 0)

    results = []
    for pred in preds:
        string = ""
        prev = -1
        for p in pred:
            p = p.item()
            if p != prev and p != blank:
                string += idx2char[p]
            prev = p
        results.append(string)
    return results

# -------------------------
# LOAD MODEL
# -------------------------
model = CRNN(num_classes=len(vocab))
model.load_state_dict(torch.load("crnn_prescription.pth", map_location="cpu"))
model.eval()

# -------------------------
# PREDICT FUNCTION
# -------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        outputs = outputs.log_softmax(2)

    pred = decode_predictions(outputs, idx2char, char2idx["<blank>"])
    return pred[0]

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print(predict_image("150.png"))
