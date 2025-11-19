`pip install -r requirements.txt

sudo apt update
sudo apt install tesseract-ocr poppler-utils

curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull llama3.1
ollama pull nomic-embed-text

ollama run llama3.2 &
ollama run llama3.1`