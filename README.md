Strictly for converting GeoChat VQA format to align with the format Florence2 was pretrained on (see Florence2 weaknesses technical documentation). TLDR from technical documentation: Florence2 hates plurals and linking unicode character that implies multiple object-categary in ```<p>phrase</p>``` such as comma, "and". <br>

Overall workflow:
1. Download GeoChat VQA training text prompts and their corresponding images from here: https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct/tree/main. If you are beginner to HuggingFace, create an account and create an API token first.
2. ```python format_GeoChat2Florence2.py``` to get VQA prompts that pretrained Florence2 get used to. Simultaneously, ```python extract_grounding.py``` to get grounding VQA prompts that pretrained Florence2 get used to.
3. ```python flatten_any_VQA.py``` to flatten both resulting datasets from format_GeoChat2Florence2.py and extract_grounding.py.
4. Familarise yourself with training loop by VScode debug debug_tuneFlorence.py. 90% of your technical questions can be resolved by spending 30 mins looking through the code a bit.
5. If run on 1 GPU, ```python debug_tuneFlorence.py```, if DDP train on multiple GPUs in 1 node, ```CUDA_VISIBLE_DEVICEs=... torchrun --nprocs_per_node=... parallel_tuneFlorence.py```

Important .py files to keep in mind.
- debug_tuneFlorence.py: Basically a training loop of Florence2 on 1 GPU that you can debug on VSCode. **Really recommend** anyone to debug the pipeline for 30 mins before asking me questions. 
- parallel_tuneFlorence.py: A scale up version of debug_tuneFlorence.py for DDP training. Some technical issues arise from manually implementing gradient accumulation with HuggingFace minibatch loss calculation but it is what it is unfortunately.
- format_GeoChat2Florence2.py: Reformat GeoChat prompt and bounding box to Florence2 expectation given insight about its weaknesses.
- extract_grounding.py: Do exactly as format_GeoChat2Florence.py does but only for ```[grounding]``` task in GeoChat Instruction dataset.
- flatten_any_VQA.py: GeoChat Instruction dataset has multiple Question-Answer pairs per Visual Question-Answer instruction sample. From looking at other repos that finetune Florence2, I notice that only 1 QA pair and 1 image being extracted per Visual Question-Answer instruction sample being input into Florence2. This means half of GeoChat Instruction prompts are being missed out per epoch unless I flatten the dataset. ***It's unknown whether original Florence2 flatten their dataset and from testing, as flattening VQA dataset seems to lead to extreme overfitting.***
