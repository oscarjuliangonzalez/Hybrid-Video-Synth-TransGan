import argparse
import os
import shutil
from pathlib import Path

import cv2
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def extract_frames_from_avi(video_path, output_folder):
    """
    Extract all frames from an AVI video file and save them as PNG images.

    Args:
        video_path (str): Path to the .avi video file
        output_folder (str): Folder where frames will be saved

    Returns:
        int: Number of frames extracted
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    video_name = Path(video_path).stem

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")

    return frame_count


def process_avi_folder(input_folder, output_base_folder):
    """
    Process all .avi files in a folder, extracting frames from each video.
    Creates a subfolder for each video's frames.

    Args:
        input_folder (str): Folder containing .avi files
        output_base_folder (str): Base folder where subfolders for each video will be created

    Returns:
        dict: Dictionary with video names as keys and frame counts as values
    """
    Path(output_base_folder).mkdir(parents=True, exist_ok=True)

    avi_files = sorted(Path(input_folder).glob("*.avi"))

    if not avi_files:
        print(f"No .avi files found in {input_folder}")
        return {}

    results = {}

    for video_file in avi_files:
        video_name = video_file.stem
        output_folder = os.path.join(output_base_folder, video_name)

        try:
            frame_count = extract_frames_from_avi(str(video_file), output_folder)
            results[video_name] = frame_count
        except Exception as exc:
            print(f"Error processing {video_file}: {exc}")
            results[video_name] = None

    print("\nProcessing complete! Summary:")
    for video_name, count in results.items():
        if count is not None:
            print(f"  {video_name}: {count} frames")
        else:
            print(f"  {video_name}: Failed")

    return results


def consolidate_images_from_subfolders(input_folder, output_folder, move=False):
    """
    Consolidate all images from a folder with subfolders into a single folder.

    This function recursively searches through all subfolders and collects all image files
    (jpg, jpeg, png, bmp, tiff, gif) into a single output folder.

    Args:
        input_folder (str): Base folder containing subfolders with images
        output_folder (str): Destination folder where all images will be collected
        move (bool): If True, move files (cut). If False, copy files. Default is False.

    Returns:
        dict: Summary with 'total_files' count and 'files_by_subfolder' breakdown
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.GIF'}

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    input_path = Path(input_folder)

    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    total_files = 0
    files_by_subfolder = {}

    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in image_extensions:
            relative_path = file_path.relative_to(input_path)
            subfolder = str(relative_path.parent)

            if subfolder not in files_by_subfolder:
                files_by_subfolder[subfolder] = 0
            files_by_subfolder[subfolder] += 1

            output_file = os.path.join(output_folder, file_path.name)

            counter = 1
            base_name = file_path.stem
            extension = file_path.suffix
            while os.path.exists(output_file):
                output_file = os.path.join(output_folder, f"{base_name}_{counter}{extension}")
                counter += 1

            if move:
                shutil.move(str(file_path), output_file)
            else:
                shutil.copy2(str(file_path), output_file)

            total_files += 1

    print(f"{'Moved' if move else 'Copied'} {total_files} image files to {output_folder}")
    if files_by_subfolder:
        print("\nFiles by subfolder:")
        for subfolder, count in sorted(files_by_subfolder.items()):
            print(f"  {subfolder}: {count} files")

    return {
        'total_files': total_files,
        'files_by_subfolder': files_by_subfolder,
        'move': move,
    }


class VQGAN(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 1),
        )

        self.codebook = nn.Embedding(vocab_size, embed_dim)
        self.codebook.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh(),
        )

    def quantize(self, z):
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, z.shape[1])

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.codebook.weight.t())

        indices = torch.argmin(d, dim=1)
        z_q = self.codebook(indices).view(z.shape[0], z.shape[2], z.shape[3], z.shape[1])
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        z_q_ste = z + (z_q - z).detach()
        return z_q_ste, indices, z_q

    def forward(self, x):
        z_e = self.encoder(x)
        z_q_ste, indices, z_q = self.quantize(z_e)
        x_hat = self.decoder(z_q_ste)
        return x_hat, indices, z_e, z_q


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        def block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride=stride, padding=1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            block(in_channels, 64),
            block(64, 128),
            block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)


class VQGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual_loss = lpips.LPIPS(net='vgg').eval()

        for param in self.perceptual_loss.parameters():
            param.requires_grad = False

    def forward(self, x, x_hat):
        return self.perceptual_loss(x, x_hat).mean()


class EchoDataset(Dataset):
    def __init__(self, folder_path, img_size=256):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('L')
        return self.transform(image)


def train_epoch(dataloader, epoch_idx, model, discriminator, opt_ae, opt_disc, perceptual_loss_fn, device):
    model.train()
    discriminator.train()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}")

    for x in pbar:
        x = x.to(device)

        opt_ae.zero_grad()

        x_hat, indices, z_e, z_q = model(x)

        rec_loss = F.l1_loss(x, x_hat)
        p_loss = perceptual_loss_fn(x, x_hat).mean()

        codebook_loss = F.mse_loss(z_e.detach(), z_q) + 0.25 * F.mse_loss(z_e, z_q.detach())

        logits_fake = discriminator(x_hat)
        g_loss = -torch.mean(logits_fake)

        total_loss_ae = rec_loss + p_loss + codebook_loss + 0.1 * g_loss

        total_loss_ae.backward()
        opt_ae.step()

        opt_disc.zero_grad()

        logits_real = discriminator(x)
        logits_fake_d = discriminator(x_hat.detach())

        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake_d))
        total_loss_disc = 0.5 * (loss_real + loss_fake)

        total_loss_disc.backward()
        opt_disc.step()

        active_tokens = len(torch.unique(indices))

        pbar.set_postfix({
            "AE_Loss": f"{total_loss_ae.item():.4f}",
            "Disc_Loss": f"{total_loss_disc.item():.4f}",
            "Tokens": active_tokens,
        })


def build_parser():
    parser = argparse.ArgumentParser(description="Train the echocardiogram VQGAN model.")
    parser.add_argument("--data-dir", default="EchoNet-Dynamic/sub all frames", help="Folder with training images.")
    parser.add_argument("--img-size", type=int, default=256, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of DataLoader workers.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--save-every", type=int, default=5, help="Checkpoint interval in epochs.")
    parser.add_argument("--checkpoint-dir", default="checkpoints_proid", help="Directory for checkpoints.")
    parser.add_argument("--samples-dir", default="samples", help="Directory for samples.")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"], help="Force device selection.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = VQGAN().to(device)
    discriminator = PatchGANDiscriminator().to(device)
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device).eval()

    opt_ae = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    dataset = EchoDataset(folder_path=args.data_dir, img_size=args.img_size)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    print(f"Iniciando entrenamiento por {args.epochs} épocas...")

    for epoch in range(1, args.epochs + 1):
        train_epoch(train_loader, epoch, model, discriminator, opt_ae, opt_disc, perceptual_loss_fn, device)

        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"vqgan_heart_ep{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'disc_state_dict': discriminator.state_dict(),
                'opt_ae_state_dict': opt_ae.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint guardado en: {checkpoint_path}")

        with torch.no_grad():
            model.eval()
            sample_x = next(iter(train_loader)).to(device)
            x_hat, _, _, _ = model(sample_x)
            sample_img = (x_hat[0] + 1) / 2
            _ = sample_img
            model.train()

    print("¡Entrenamiento completado!")


if __name__ == "__main__":
    main()
