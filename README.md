***Obv, I can't say much in detail about how deal with each individual dataset because NDA*** But here I can share some common senses to preprocess any VQA in general for any VLM. Here's an example to reformat some random VQA to align with the format Florence2 was pretrained on (see Florence2 weaknesses technical documentation).

Overall workflow:
1. Download GeoChat VQA training text prompts and their corresponding images from here: https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct/tree/main. If you are beginner to HuggingFace, create an account and create an API token first.
2. ```python format_GeoChat2Florence2.py``` to get VQA prompts that pretrained Florence2 get used to. 
3. ```python flatten_any_VQA.py``` to flatten both resulting datasets from format_GeoChat2Florence2.py and extract_grounding.py.
4. Familarise yourself with training loop by VScode debug debug_tuneFlorence.py. 90% of your technical questions can be resolved by spending 30 mins looking through the code a bit.
5. If run on 1 GPU, ```python debug_tuneFlorence.py```, if DDP train on multiple GPUs in 1 node, ```CUDA_VISIBLE_DEVICEs=... torchrun --nprocs_per_node=... parallel_tuneFlorence.py```

Important .py files to keep in mind.
- debug_tuneFlorence.py: Basically a training loop of Florence2 on 1 GPU that you can debug on VSCode. **Really recommend** anyone to debug the pipeline for 30 mins before asking me questions. 
- parallel_tuneFlorence.py: A scale up version of debug_tuneFlorence.py for DDP training. Some technical issues arise from manually implementing gradient accumulation with HuggingFace minibatch loss calculation but it is what it is unfortunately.
- format_prompt.py: See VQA prompt preprocessing.pdf
- flatten_any_VQA.py: Some VQA has many Question-Answer pairs per instruction sample. From looking at other repos that finetune Florence2, I notice that only 1 QA pair and 1 image being extracted per instruction sample being input into Florence2. This means 40-50% of Question-Answer pairs are being ignroed per epoch unless I flatten the dataset.
