# owod_inference.py
# OWOD (Open World Object Detection) 추론 템플릿
# 원본 논문 및 코드 기반: https://github.com/JosephKJ/OWOD

import torch
import torchvision.transforms as T
from PIL import Image
from models.owod_model import OWODModel  # OWOD 논문 기반 custom model class

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기 (사전학습된 모델 경로 지정)
def load_owod_model(model_path="owod_checkpoint.pth"):
    model = OWODModel(num_known_classes=20)  # 예: COCO의 일부 클래스만 known
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    return model

# 이미지 전처리
def preprocess_image(image: Image.Image):
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# 추론 실행
def run_owod_inference(image_path, model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)[0]

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    known_classes = model.get_known_class_names()
    results = []

    for box, score, label in zip(boxes, scores, labels):
        class_name = known_classes[label] if label < len(known_classes) else "Unknown"
        results.append({
            "class": class_name,
            "score": float(score),
            "box": box.tolist()
        })

    return results