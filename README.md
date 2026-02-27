# ONNX Models

**Principais modelos**

- [`SimianLuo/LCM_Dreamshaper_v7`](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7#latent-consistency-models)  (SD 1.5 + LCM, ultra‑rápido com poucos steps)
- [`etri-vilab/koala-700m`](https://huggingface.co/etri-vilab/koala-700m)  (SDXL comprimido, rápido em GPU)
- [`FFusion/FFusionXL-BASE`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (SDXL base)
- [`stabilityai/sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo)(variante Turbo)

---

## Passo a Passo

1. Criar e Ativar venv
    1. Windows:
        
        ```python
        python -m venv .venv
        . .\\.venv\\Scripts\\Activate.ps
        ```
        
    2. Linux
        
        ```python
        python3 -m venv .venv
        source .venv/bin/activate
        ```
        
2. Instalar Depedências
    - Em caso de erros, mesmo com as depedências
    
    ```python
    pip install --upgrade diffusers transformers accelerate huggingface_hub[cli] safetensors onnx numpy
    ```
    
3. (Opcional) Autenticar no Hugging Face
    
    ```python
    hf auth login
    # ou
    huggingface-cli login
    ```
    
    - Cole seu **token** (criado em [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

### Como utilizar

Os modelos estão divididos pelo nome e se são ou não ONNX. Você deve escolher qual modelo quer e utilizar o comando:

```python
py [nome_do_modelo].py
#ou
python[nome_do_modelo].py
```

Os modelo irá ser baixado na primeira execução, e a imagem será gerada em seguida, em execuções após a primeira, apenas gerara a imagem sem baixar o modelo novamente.

Os resultados da execução atual são mostrados no terminal ao final da execução + armazenados na pasta logs, as imagens geradas são armazenadas na pasta images.
