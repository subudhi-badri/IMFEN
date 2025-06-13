import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
import yaml
import argparse

from models.mfen import MFEN
from datasets.uieb_dataset import UIEBDataset, UIEBUnpairedDataset
from losses.perceptual_loss import MultiVGGPerceptualLoss
from utils.metrics import PSNR, SSIM

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def train(epoch, train_dataloader, optimizer, model, loss_fn, writer, num_batches, device):
    tbar = tqdm(train_dataloader)
    total_loss = 0.
    total = 0
    model.train()

    for batch_index, batch in enumerate(tbar):
        gt = batch['label'].to(device)
        in_img = batch['in_img'].to(device)

        optimizer.zero_grad()

        out1, out2, out3 = model(in_img)
        loss = loss_fn(out1, out2, out3, gt)

        loss.backward()
        optimizer.step()

        total += 1
        total_loss += loss.item()
        avg_loss = total_loss / total

        if batch_index % 10 == 0:
            iters = batch_index + epoch * num_batches
            writer.add_scalar("Train/Loss", avg_loss, iters)

        tbar.set_description("[Epoch {}] [Avg loss : {:.4f}]".format(epoch + 1, avg_loss))
        tbar.update()

def validate(epoch, model, val_dataloader, psnr_model, ssim_model, results_dir, device):
    model.eval()

    with torch.no_grad():
        total_psnr = 0.
        total_ssim = 0.
        count = 0.
        epoch_result_dir = os.path.join(results_dir, f"epoch_{epoch + 1}")
        make_directory(epoch_result_dir)

        for batch_index, batch in enumerate(val_dataloader):
            gt = batch['label'].to(device)
            in_img = batch['in_img'].to(device)

            restored, out_2, out_3 = model(in_img)

            psnr = psnr_model(restored, gt)
            ssim = ssim_model(restored, gt)

            total_psnr += psnr.mean().item()
            total_ssim += ssim.mean().item()
            count += 1

            in_img = in_img.detach().cpu()
            restored_img = restored.detach().cpu()
            gt_img = gt.detach().cpu()

            for i in range(len(batch['filename'])):
                save_image(restored_img[i], os.path.join(epoch_result_dir, batch['filename'][i]))

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count

        return avg_psnr, avg_ssim

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create directories
    samples_dir = os.path.join("samples", config['exp_name'])
    results_dir = os.path.join("results", config['exp_name'])
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create model
    model = MFEN(
        en_feature_num=config['model']['en_feature_num'],
        en_inter_num=config['model']['en_inter_num'],
        de_feature_num=config['model']['de_feature_num'],
        de_inter_num=config['model']['de_inter_num']
    ).to(device)

    # Create datasets
    train_dataset = UIEBDataset(
        train_dataset=os.path.abspath(config['train_dataset']),
        crop_size=config['crop_size'],
        test_dataset=os.path.abspath(config['test_dataset'])
    )

    val_dataset = UIEBDataset(
        train_dataset=os.path.abspath(config['train_dataset']),
        crop_size=config['crop_size'],
        test_dataset=os.path.abspath(config['test_dataset']),
        mode='valid'
    )

    test_dataset = UIEBUnpairedDataset(
        train_dataset=os.path.abspath(config['train_dataset']),
        crop_size=config['crop_size']
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )

    # Create optimizer and loss function
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'initial_lr': config['learning_rate']}],
        betas=(0.9, 0.999)
    )
    loss_fn = MultiVGGPerceptualLoss(
        lam=config['loss']['lam'],
        lam_p=config['loss']['lam_p'],
        lam_lpips=config['loss']['lam_lpips']
    ).to(device)

    # Create metrics
    psnr_model = PSNR(
        crop_border=config['metrics']['crop_border'],
        only_test_y_channel=config['metrics']['only_test_y_channel'],
        data_range=config['metrics']['data_range']
    ).to(device)
    
    ssim_model = SSIM(
        crop_border=config['metrics']['crop_border'],
        only_test_y_channel=config['metrics']['only_test_y_channel'],
        data_range=config['metrics']['data_range'] * 255.0
    ).to(device)

    # Create tensorboard writer
    writer = SummaryWriter(os.path.join("samples", "logs", config['exp_name']))

    # Training loop
    num_batches = len(train_dataloader)
    start_epoch = 0
    best_psnr = 0.
    best_ssim = 0.

    if config['resume']:
        checkpoint = torch.load(config['resume'])
        start_epoch = checkpoint["epoch"]
        best_psnr = max(best_psnr, checkpoint["psnr"])
        best_ssim = max(best_ssim, checkpoint["ssim"])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded checkpoint with epoch {start_epoch}, PSNR {checkpoint['psnr']:.4f} and SSIM {checkpoint['ssim']:.4f}")

    for epoch in range(start_epoch, config['epochs']):
        # Train epoch
        train(epoch, train_dataloader, optimizer, model, loss_fn, writer, num_batches, device)

        # Validate epoch
        psnr, ssim = validate(epoch, model, val_dataloader, psnr_model, ssim_model, results_dir, device)

        # Log metrics
        writer.add_scalar("Validation/PSNR", psnr, epoch + 1)
        writer.add_scalar("Validation/SSIM", ssim, epoch + 1)

        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)

        print(f"[Epoch {epoch + 1}] [Avg PSNR: {psnr:.4f}] [Avg SSIM: {ssim:.4f}]")

        # Save checkpoint
        save_path = os.path.join(samples_dir, f"epoch_{epoch + 1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "psnr": psnr,
            "ssim": ssim,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, save_path)

    print(f"Best PSNR: {best_psnr:.4f} and SSIM: {best_ssim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args) 
