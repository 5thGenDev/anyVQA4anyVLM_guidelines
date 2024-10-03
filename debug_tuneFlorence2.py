import os
import math
import torch
import torch.distributed as dist

from typing import Tuple
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from transformers import (AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoTokenizer, 
                          get_scheduler)
from PIL import Image
import json
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()


processor = AutoProcessor.from_pretrained("./Florence-2-large/", trust_remote_code=True, local_files_only=True)


class GeoFlorence2_data(Dataset):
    # class CustomDataset is first instantiated, before being wrapped with DistributedSampler
    def __init__(self, data_path: str, images_path: str):
        super(GeoFlorence2_data, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict = list_data_dict
        self.images_path = images_path
        del list_data_dict
        flush()
        
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i): 
        image_file, human_question, lm_answer = self.list_data_dict[i] 
        image_path = os.path.join(self.images_path, image_file)
        image = Image.open(image_path)   
        return human_question, lm_answer, image


def collate_fn(batch):
    questions, answers, images = zip(*batch)    # return batch input as tuple
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True)
    return inputs, answers


def load_model_processor(output_dir : str):
    r""" Debug documentation. In case someone else debugs this 
    Inputs: 
        - output_dir = '~/path/to/model_checkpoints/'. Expect epoch_*/ subfolders under model_checkpoints
        - global processor: transformers_modules.processing_florence2.Florence2Processor
        - global processor.tokenizer = transformers.models.bart.tokenization_bart_fast.BartTokenizerFast
    
    Outputs: 
        - model: transformers_modules.modeling_florence2.Florence2ForConditionalGeneration (pretrained or from checkpoint)
        - latest_epoch: int
        - global processor: transformers_modules.processing_florence2.Florence2Processor
        - global processor.tokenizer = transformers.models.bart.tokenization_bart_fast.BartTokenizerFast
        
    Goals:
        - Load pretrained weights from resuming checkpoint to florence2 model. 
        - Set model.config.vision_config.model_type = "davit"
        - Add new special tokens to processor.tokenizer if training from scratch, 
            - or have processor.tokenizer load vocabulary from checkpoint if resume training.

    DONT:
        - Load processor from checkpoint: 'processor=AutoProcessor(checkpoint_dir)
    """
    
    global processor
    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith("epoch_")]
    if not checkpoint_files:
        latest_epoch = 0
        model = AutoModelForCausalLM.from_pretrained("./Florence-2-large/", 
                                                     trust_remote_code=True, 
                                                     local_files_only=True)
        tokenizer = processor.tokenizer
        tokens_to_add = {
            'additional_special_tokens': \
                tokenizer.additional_special_tokens + \
                ['<od>', '</od>', '<ocr>', '</ocr>'] + \
                [f'<loc_{x}>' for x in range(1000)] + \
                ['<cap>', '</cap>', '<ncap>', '</ncap>','<dcap>', '</dcap>', '<grounding>', '</grounding>', '<seg>', '</seg>', '<sep>', '<region_cap>', '</region_cap>', '<region_to_desciption>', '</region_to_desciption>', '<proposal>', '</proposal>', '<poly>', '</poly>', '<and>'] + \
                ['<p>', '</p>', '<delim>'] + \
                [f'<angle_{x}>' for x in range(360)]
        }
        tokenizer.add_special_tokens(tokens_to_add)  
        processor.tokenizer = tokenizer
        model.resize_token_embeddings(len(processor.tokenizer))
        print("Starting training from scratch.")
    else: 
        latest_epoch = max(int(f.split('_')[-1]) for f in checkpoint_files)
        checkpoint_dir = os.path.join(output_dir, f"epoch_{latest_epoch}/")
        model_config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        if model_config.model_type == "florence2":
            model_config.vision_config.model_type = "davit"
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, 
                                                     trust_remote_code=True, 
                                                     local_files_only=True, 
                                                     use_safetensors=True, 
                                                     config=model_config)
        print(f"Get safetensors pretrained weights from {checkpoint_dir}")
        
        chpk_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        processor.tokenizer = chpk_tokenizer
        print(f"Resume training from epoch {latest_epoch}")
    
    return model, latest_epoch


def load_training_components(
    based_global_batch, 
    batch_per_gpu, 
    epochs, 
    latest_epoch, 
    output_dir, 
    warmup_ratio, 
    train_loader, 
    model, 
    num_gpus, 
    lr
):
    accumulate_steps = based_global_batch // (num_gpus * batch_per_gpu)

    # Effective batch size = based_global_batch = accumulate_steps * num_nodes * num_gpu * batch_per_gpu
    total_LR = lr * math.sqrt(based_global_batch // batch_per_gpu)
    num_training_steps = (epochs - latest_epoch) * len(train_loader) // accumulate_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=total_LR)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps*warmup_ratio),        # chunk of num_training_steps for increasing total_LR
        num_training_steps=num_training_steps,
    )
    if latest_epoch == 0:
        return accumulate_steps, optimizer, lr_scheduler
    
    optimizer_scheduler_checkpoint = torch.load(f"{output_dir}/epoch_{latest_epoch}/optimizer_scheduler.pth")
    optimizer.load_state_dict(optimizer_scheduler_checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(optimizer_scheduler_checkpoint['scheduler_state_dict'])
    
    return accumulate_steps, optimizer, lr_scheduler


def print_number_of_tokens(step, input_ids, labels, padding_token_id, model):
    valid_tokens_per_sequence = ((input_ids.view(-1, model.config.vocab_size) != padding_token_id) & (labels.view(-1) != padding_token_id)).sum(dim=1)
    total_tokens = valid_tokens_per_sequence.sum().item()
    print(f"Mini-batch {step + 1} has {total_tokens} tokens")
    return total_tokens


def train_model(
    model, 
    processor, 
    train_loader, 
    output_dir, 
    device, 
    latest_epoch, 
    epochs, 
    based_global_batch, 
    batch_per_gpu, 
    warmup_ratio, 
    lr
):
    # see explanation here: https://github.com/Lightning-AI/pytorch-lightning/discussions/3706#discussioncomment-900300
    num_gpus = 1
    epochs = latest_epoch + epochs
    accumulate_steps, optimizer, lr_scheduler = load_training_components(
        based_global_batch, 
        batch_per_gpu, 
        epochs, 
        latest_epoch, 
        output_dir, 
        warmup_ratio, 
        train_loader, 
        model, 
        num_gpus, 
        lr
    )

    for epoch in range(latest_epoch, epochs):
        epoch_loss = 0
        grad_accum_loss = 0
        accumulated_num_tokens = 0
        for step, (inputs, answers) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(device)
            num_tokens = print_number_of_tokens(step, input_ids, labels, processor.tokenizer.pad_token_id, model)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss = loss / accumulate_steps
            if accumulate_steps > 1:
                loss = loss * num_tokens
                accumulated_num_tokens += num_tokens
                    
            del inputs, answers
            flush()
            loss.backward()
            
            print(f"Minibatch {step + 1} loss is: {loss.detach().item()}")
            epoch_loss = epoch_loss + loss.detach().item()
            grad_accum_loss = grad_accum_loss + loss.detach().item()
            if (step + 1) % accumulate_steps == 0:    
                print(f"Gradient Accumulation Loss: {grad_accum_loss}") 
                print(f"Accumulated tokens: {accumulated_num_tokens}") 
                print(f"Computed loss: {grad_accum_loss / accumulated_num_tokens}")
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad / accumulated_num_tokens
    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)  
                continue
            
        epoch_loss = epoch_loss.item() / num_gpus
        epoch_loss_per_sample = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}: Average Training Loss: {epoch_loss}\n")
        print(f"Epoch {epoch+1}: Average Training Loss per VQA sample: {epoch_loss_per_sample}\n")
        checkpoint_dir = f"epoch_{epoch+1}"
        checkpoint_dir = os.path.join(output_dir,checkpoint_dir)
        
        model.module.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
        }, f"{checkpoint_dir}/optimizer_scheduler.pth")
        
        del epoch_loss, epoch_loss_per_sample, checkpoint_dir
        flush() # Only useful after using del, nothing more.
            

def main():
    text_path = "flatten_Florence_Instruct.json"
    images_path = "./share/softwares/kartik/GeoChat_finetuning/final_images_llava"
    output_dir = "./model_checkpoints"
    epochs = 1
    batch_per_gpu = 2
    based_global_batch = 6
    warmup_ratio = 0.03
    worker_per_gpu = 2
    lr = 2e-5
    
    train_dataset = GeoFlorence2_data(data_path=text_path, images_path=images_path)
    
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model, latest_epoch = load_model_processor(output_dir)
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_per_gpu, 
                              shuffle=False,            # false if using DistributedSampler
                              collate_fn=collate_fn, 
                              num_workers=worker_per_gpu,
                              persistent_workers=True,  # __getitem__ has no code causing memory leak, so this is fine!
                              pin_memory=True)          # need because loading batch data from CPU -> GPU
    
    train_model(model=model, 
                processor=processor,
                train_loader=train_loader, 
                output_dir=output_dir, 
                latest_epoch=latest_epoch, 
                epochs=epochs, 
                lr=lr, 
                warmup_ratio=warmup_ratio, 
                batch_per_gpu=batch_per_gpu, 
                based_global_batch=based_global_batch, 
                device=device)

    flush()
    

if __name__ == '__main__':
    main()
