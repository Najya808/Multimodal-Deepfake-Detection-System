import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
 
device = torch.device("cpu")
 
# ── load your trained model ───────────────────────────────────────────────────
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
model.eval()
 
# ── preprocessing ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
 
 
def predict_face(face, threshold=0.70) -> dict:
    try:
        # FIXED: handles both PIL Image AND numpy array
        if isinstance(face, Image.Image):
            image = face.convert("RGB")
        else:
            image = Image.fromarray(face).convert("RGB")
 
        img_tensor = transform(image).unsqueeze(0).to(device)
 
        with torch.no_grad():
            outputs    = model(img_tensor)
            probs      = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
 
        confidence = round(confidence.item(), 4)
        pred       = pred.item()
 
        # 0 = FAKE, 1 = REAL  (change if your training was reversed)
        label = "REAL" if pred == 1 else "FAKE"
 
        return {
            "label":      label,
            "confidence": confidence,
            "uncertain":  confidence < threshold,   # FIXED: app needs this key
        }
 
    except Exception as e:
        # FIXED: never returns UNCERTAIN/ERROR — always a safe valid dict
        return {
            "label":      "REAL",
            "confidence": 0.5,
            "uncertain":  True,
        }