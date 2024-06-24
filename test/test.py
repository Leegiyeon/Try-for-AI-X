import torch
from transformers import pipeline

# GPU가 사용 가능한지 확인 (Metal 지원)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device")
else:
    device = torch.device('cpu')
    print("Using CPU device")

# 모델 로드
classifier = pipeline('sentiment-analysis')

# 모델을 GPU로 이동
classifier.model.to(device)

# 텍스트 분석
text = "난 Hugging Face 서비스를 사용할 준비를 하고 있는데 설레!"

# 텍스트를 토큰화하고 입력 텐서를 GPU로 이동
inputs = classifier.tokenizer(text, return_tensors='pt').to(device)

# 모델에 입력을 전달하고 결과를 GPU에서 CPU로 이동
with torch.no_grad():
    outputs = classifier.model(**inputs)
    outputs = outputs.logits.cpu()

# 결과 후처리
predictions = torch.nn.functional.softmax(outputs, dim=-1)
labels = classifier.model.config.id2label
result = [{"label": labels[i], "score": predictions[0][i].item()} for i in range(len(labels))]
print(result)
