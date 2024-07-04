To zip and create layer:

    python -m venv venv
    source venv/bin/activate

    pip install huggingface_hub langchain-core langchain-huggingface langchain-text-splitters 

    pip freeze > requirements.txt

    mkdir python
    pip install -r requirements.txt -t python/

    mkdir python
    pip install -r requirements.txt -t python/
    zip -r9 layer.zip python
    