"""
Communicating between GPUs is always slower than doing redundant process in each asynch GPU.
"""

import os
import math
import torch
import torch.distributed as dist
from datetime import timedelta

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import sys

from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, AutoTokenizer, get_scheduler
from PIL import Image
import json
import gc
import multiprocessing
import argparse
import signal
import re

# where all datasets and pretrained florence2 repo is in: "/home/ubuntu/Documents/malik/datasets/satellite/GeoChat_images"
def no_indents_and_line_breaks_str(sample_string: str):
    sample_string = re.sub(r'^[ \t]+', '', sample_string, flags=re.MULTILINE)
    return sample_string.replace("\n", "").strip()


def flush():
    gc.collect()
    torch.cuda.empty_cache()


def signal_handler(sig, frame):
    print("Signal received, terminating processes...")
    flush()
    sys.exit(0)


processor = AutoProcessor.from_pretrained(
    "./Florence-2-large/", trust_remote_code=True, local_files_only=True
)


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))  
    world_size = int(os.environ.get("WORLD_SIZE", 1)) 
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    original_gpu_id = int(cuda_visible_devices[local_rank])
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=4320),
    )  
    sample_string = f"""Process group initialized. 
          Local rank: {local_rank}, 
          Global Rank: {rank}, 
          World size: {world_size}, 
          Original GPU id: {original_gpu_id}"""
    print(no_indents_and_line_breaks_str(sample_string))
    dist.barrier()
    return local_rank


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with given arguments."
    )
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="/absolute/path/to/VQA.json or relative/path/to/VQA.json",
    )
    parser.add_argument(
        "--images_path",
        type=str,
        required=True,
        help="/absolute/path/to/image_folder or relative/path/to/image_folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="/absolute/path/to/checkpoint_folder or relative/path/to/checkpoint_folder",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="num_epochs w.r.t to model checkpoint epoch. Was 3 for 280k GeoChat finetuning.",
    )
    parser.add_argument(
        "--batch_per_gpu",
        type=int,
        default=2,
        help="CUDA OOM at 92% training with batch size = 6.",
    )
    parser.add_argument(
        "--based_global_batch",
        type=int,
        default=160,
        help="Nominal global batch for GeoChat finetuning was 144.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for GeoChat finetuning was 0.03.",
    )
    parser.add_argument(
        "--worker_per_gpu",
        type=int,
        default=4,
        help="Num_worker for dataloader CPU, best seems to be 4 or 5",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Base LR for GeoChat finetuning was 2e-5.",
    )

    return parser.parse_args()


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
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    )
    return inputs, answers


def train_model(
    model,
    processor,
    train_loader,
    output_dir,
    latest_epoch,
    epochs,
    warmup_ratio,
    based_global_batch,
    batch_per_gpu,
    lr,
    device
):
    epochs += latest_epoch 
    accumulate_steps, lr_scheduler, optimizer, num_training_steps = load_training_components(model,
                                                                                            train_loader, 
                                                                                            output_dir,
                                                                                            latest_epoch, 
                                                                                            epochs, 
                                                                                            warmup_ratio, 
                                                                                            based_global_batch, 
                                                                                            batch_per_gpu, 
                                                                                            lr)
        
    # output_dir = f"{output_dir}_grounding"      # uncomment this when finetuning grounding-only
    epoch_loss = torch.tensor(0.0)
    chunk_loss = torch.tensor(0.0)
    for epoch in range(latest_epoch, epochs):
        log_file = f"{output_dir}/logs/epoch_{epoch + 1}_loss_log.txt"
        with open(log_file, 'w') as f:
            f.write(f"{num_training_steps} Total training step.\n")
        
        for step, (inputs, answers) in enumerate(
            tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        ):
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)
            
            # Gradient accumulation only during intermediate steps 
            if step % accumulate_steps != 0:
                with model.no_sync():
                    outputs = model(
                        input_ids=input_ids, pixel_values=pixel_values, labels=labels
                    )
                    loss = outputs.loss
                    loss = loss / accumulate_steps
                    loss.backward()  
            
            else:
                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss
                loss = loss / accumulate_steps
                loss.backward()   
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
         
            if step == 0:
                epoch_loss.to(loss.detach().dtype)
                chunk_loss.to(loss.detach().dtype)
            epoch_loss = epoch_loss + loss.detach()
            chunk_loss = chunk_loss + loss.detach()
            
            if step % (1000 * accumulate_steps // (epochs - latest_epoch)) == 0:
                dist.all_reduce(chunk_loss, op=dist.ReduceOp.AVG)
                if dist.get_rank() == 0:
                    log_chunk_loss = chunk_loss.item()
                    log_file = f"{output_dir}/logs/epoch_{epoch + 1}_loss_log.txt"
                    with open(log_file, 'a') as f:
                        f.write(f"Total chunk loss per 1000 training steps = {log_chunk_loss}\n")
                    chunk_loss.zero_()

        dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)
        if dist.get_rank() == 0:
            log_epoch_loss = epoch_loss.item()
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
            model.module.save_pretrained(checkpoint_dir)
            processor.tokenizer.save_pretrained(checkpoint_dir)
            torch.save(
                {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                },
                f"{checkpoint_dir}/optimizer_scheduler.pth",
            )
            log_file = f"{output_dir}/logs/epoch_{epoch + 1}_loss_log.txt"
            with open(log_file, 'a') as f:
                f.write(f"Total loss entire epoch: {log_epoch_loss}\n")
            
            epoch_loss.zero_()
            del checkpoint_dir
            flush()  # Only useful after using del, nothing more.
            torch.cuda.memory_summary(device=None, abbreviated=False)
        
    del model, processor, train_loader, optimizer, lr_scheduler
    flush()


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
        - Add new tokens to checkpoint vocabularies.
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
                [f'<angle_{x}>' for x in range(360)]
        }
        tokenizer.add_special_tokens(tokens_to_add)  
        processor.tokenizer = tokenizer
        model.resize_token_embeddings(len(processor.tokenizer))
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
        chpk_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        processor.tokenizer = chpk_tokenizer
    
    return model, latest_epoch


def load_training_components(
    model,
    train_loader,
    output_dir,
    latest_epoch,
    epochs,
    warmup_ratio,
    based_global_batch,
    batch_per_gpu,
    lr
):
    accumulate_steps = based_global_batch // (dist.get_world_size() * batch_per_gpu)
    scaled_LR = lr * math.sqrt(based_global_batch // batch_per_gpu)
    num_training_steps = ((epochs-latest_epoch) * len(train_loader) // accumulate_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=scaled_LR)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps*warmup_ratio),
        num_training_steps=num_training_steps,
    )
    if latest_epoch > 0:
        optimizer_scheduler_checkpoint_path = os.path.join(output_dir, f"epoch_{latest_epoch}/optimizer_scheduler.pth")
        optimizer_scheduler_checkpoint = torch.load(optimizer_scheduler_checkpoint_path)
        optimizer.load_state_dict(
            optimizer_scheduler_checkpoint["optimizer_state_dict"]
        )
        lr_scheduler.load_state_dict(
            optimizer_scheduler_checkpoint["scheduler_state_dict"]
        )
        
    return accumulate_steps, lr_scheduler, optimizer, num_training_steps


def worker_function(args):
    local_rank = setup_distributed()
    
    train_dataset = GeoFlorence2_data(
        data_path=args.text_path, images_path=args.images_path
    )

    model, latest_epoch = load_model_processor(args.output_dir)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    # Essential for loss.backward, optimizer.step in DDP training.

    # DistributedSampler follows global rank so each sampler replica holds a number of indices that amounts to dataset/num_replicas
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    # DONT set batch_size=batch_per_gpu*num_gpu, num_workers=worker_per_gpu*num_gpu
    # shuffle=False if DistributedSampler already has shuffle=True
    # persistent_workers=True if class YourDataset(Datasets).__getitem__ has no code causing memory leak
    # pin_memory=True always in DDP!
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_per_gpu,
        shuffle=False,  
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.worker_per_gpu,
        persistent_workers=True,  
        pin_memory=True,
    )

    train_model(
        model=model,
        processor=processor,
        train_loader=train_loader,
        output_dir=args.output_dir,
        latest_epoch=latest_epoch,
        epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        based_global_batch=args.based_global_batch,
        batch_per_gpu=args.batch_per_gpu,
        lr=args.lr,
        device=local_rank
    )

    flush()
    dist.destroy_process_group()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    multiprocessing.set_start_method("spawn", force=True)
    args = parse_arguments()

    # Start your multiprocessing work
    process = multiprocessing.Process(target=worker_function, args=(args,))
    process.start()
    try:
        process.join()
    except KeyboardInterrupt:
        dist.destroy_process_group()
        flush()
        print("Interrupted by user")
        process.terminate()
        process.join()
