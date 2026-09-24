import modal

app = modal.App("selectiveqa-check")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")

@app.function(gpu="T4", image=image)
def check():
    import torch
    print(torch.cuda.get_device_name(0), torch.cuda.is_available())

@app.local_entrypoint()
def main():
    check.remote()