import os
import argparse
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
from diffusion_ffpe.model import Diffusion_FFPE, initialize_text_encoder
from diffusion_ffpe.my_utils import build_transform
from torch.cuda.amp import autocast

Image.MAX_IMAGE_PIXELS = 10000000000  # 10 billion pixels
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_args_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default="TEST_FF_PATH")
    parser.add_argument('--model_path', type=str, default="stabilityai/sd-turbo")
    parser.add_argument('--pretrained_path', type=str, default="./checkpoints/model.pkl")
    parser.add_argument('--output_path', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--prompt', type=str, default="paraffin section")
    parser.add_argument('--image_prep', type=str, default='no_resize', help='the image preparation method')
    parser.add_argument('--direction', type=str, default='a2b')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images to process per batch')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()
    return args

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform, normalize_transform):
        self.paths = glob.glob(os.path.join(img_dir, '*'))
        self.transform = transform
        self.normalize_transform = normalize_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        input_image = Image.open(path).convert('RGB')
        input_img = self.transform(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = self.normalize_transform(x_t)
        # Return the tensor, original width/height for resizing back, and filename
        return x_t, input_image.width, input_image.height, os.path.basename(path)

def main(args):
    # make output folder
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving outputs to: {args.output_path}")

    # initialize the model
    model = Diffusion_FFPE(pretrained_path=args.pretrained_path, model_path=args.model_path,
                           enable_xformers_memory_efficient_attention=True)
    model = model.half().cuda()
    model.eval()

    tokenizer, text_encoder = initialize_text_encoder(args.model_path)
    a2b_tokens = tokenizer(args.prompt, max_length=tokenizer.model_max_length, padding="max_length",
                           truncation=True, return_tensors="pt").input_ids[0]
    text_emb = text_encoder(a2b_tokens.unsqueeze(0).cuda())[0].detach().cuda()

    T_val = build_transform(args.image_prep)
    normalize_transform = transforms.Normalize([0.5], [0.5])

    # Create dataset and dataloader for batching
    dataset = ImageDataset(args.img_path, T_val, normalize_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Inference in batches
    with torch.no_grad(), autocast():
        for x_t, widths, heights, filenames in tqdm(dataloader):
            x_t = x_t.cuda(non_blocking=True)
            output = model(x_t, direction=args.direction, text_emb=text_emb)
            output = output * 0.5 + 0.5  # Convert back to [0,1]

            # Save each image in the batch
            for out_img, w, h, fname in zip(output, widths, heights, filenames):
                output_pil = transforms.ToPILImage()(out_img.cpu())
                output_pil = output_pil.resize((w.item(), h.item()))
                save_path = os.path.join(args.output_path, fname)
                output_pil.save(save_path)

if __name__ == "__main__":
    inference_args = parse_args_inference()
    main(inference_args)
