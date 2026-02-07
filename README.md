# ComfyUI-TP-IMtalker
Comfy UI nodes for IMtalker to run native weights 


![imtalker](https://github.com/user-attachments/assets/d2a0eade-bab2-45b4-85eb-81123e2a35aa)

✅ Install on Comfy via Manager: 

![imtalker2](https://github.com/user-attachments/assets/729a90c8-928a-43bd-ae63-9d14d1de8a2b)


✅ Install on Comfy manually: 


```
cd custom_nodes
git clone https://github.com/tpc2233/ComfyUI-TP-IMtalker.git

cd ComfyUI-TP-IMtalker
pip install -r requirements.txt
```


✅ Comfy Workflow:
```
https://github.com/tpc2233/ComfyUI-TP-IMtalker/blob/main/workflow_imtalker.json
```

✅ Autodownload the models:

It will download the models first time you run workflow



✅ Run and Tested on::

17Gb VRAM usage max for both Audio and Video driven

RTX 6000 Pro  

--pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 

Conda env python=3.12.9 -y


Original repo:  
https://github.com/cbsjtu01/IMTalker
