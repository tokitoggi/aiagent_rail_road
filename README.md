# aiagent_rail_road

## 1. 로컬 환경 설정 
1. 파이썬 가상환경 만들기 (선택 권장)
```
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
```
2. 필수 패키지 설치
```
    pip install streamlit openai transformers torch torchvision pillow pandas
```
- sqlite3는 파이썬 내장 모듈이라 설치 필요 없음.

3. 앱 실행
```
    streamlit run app.py
```
- 브라우저가 자동으로 열리고 http://localhost:8501에서 웹앱이 실행됨!
