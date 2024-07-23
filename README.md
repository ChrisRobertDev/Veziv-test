# Veziv-test

Proiectul folosește librăria Unsloth pentru a putea antrena un model mic pe un T4 din cadrul free din Google Colab.

### Rularea proiectului

1. **Descarcarea dependințelor:**
   ```bash
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

2. Descarcarea modelului gemma-2-9b , cuantizat in 4 biti

3. Adaugarea adaptorilor Lora in asa fel incat sa finetunezi pana in 10% din parametri

4. Pregatirea datasetului in format json

5. Antrenarea modelului

6 Rularea unui exemplu
