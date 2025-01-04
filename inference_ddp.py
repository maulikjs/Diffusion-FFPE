import os
import argparse
import glob
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from torch.amp import autocast

from diffusion_ffpe.model import Diffusion_FFPE, initialize_text_encoder
from diffusion_ffpe.my_utils import build_transform

Image.MAX_IMAGE_PIXELS = 10000000000
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_args_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="stabilityai/sd-turbo")
    parser.add_argument('--pretrained_path', type=str, default="./checkpoints/model.pkl")
    parser.add_argument('--output_path', type=str, default='output', help='directory to save outputs')
    parser.add_argument('--prompt', type=str, default="paraffin section")
    parser.add_argument('--image_prep', type=str, default='no_resize')
    parser.add_argument('--direction', type=str, default='a2b')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.paths = glob.glob(os.path.join(img_dir, '*'))
        self.paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        input_image = Image.open(path).convert('RGB')
        processed_img = self.transform(input_image)
        x_t = transforms.ToTensor()(processed_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t)
        return x_t, input_image.width, input_image.height, os.path.basename(path)

def main():
    args = parse_args_inference()

    # Initialize the default process group
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)

    # Load model
    model = Diffusion_FFPE(pretrained_path=args.pretrained_path, model_path=args.model_path,
                           enable_xformers_memory_efficient_attention=True)
    model = model.half().to(rank)
    model.eval()

    # Initialize text encoder and tokenizer
    tokenizer, text_encoder = initialize_text_encoder(args.model_path)
    a2b_tokens = tokenizer(args.prompt, max_length=tokenizer.model_max_length,
                           padding="max_length", truncation=True, return_tensors="pt").input_ids[0].to(rank)

    T_val = build_transform(args.image_prep)

    dataset = ImageDataset(args.img_path, T_val)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    dist.barrier()
    progress = tqdm(dataloader, desc=f"Rank {rank}", disable=(rank != 0))

    with torch.no_grad(), autocast('cuda'):
        for x_t, widths, heights, filenames in progress:
            x_t = x_t.to(rank, non_blocking=True)
            B = x_t.shape[0]

            # Replicate tokens to match batch size
            tokens_batch = a2b_tokens.unsqueeze(0).repeat(B, 1)
            text_emb = text_encoder(tokens_batch)[0].detach().to(rank)  # [B, seq_len, hidden_dim]

            output = model(x_t, direction=args.direction, text_emb=text_emb)
            output = (output * 0.5 + 0.5).clamp(0, 1)  # ensure output in [0,1]

            for out_img, w, h, fname in zip(output, widths, heights, filenames):
                output_pil = transforms.ToPILImage()(out_img.cpu())
                output_pil = output_pil.resize((w.item(), h.item()))
                save_path = os.path.join(args.output_path, fname)
                if rank == 0:
                    output_pil.save(save_path)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
