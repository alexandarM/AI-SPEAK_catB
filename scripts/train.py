"""
scripts/train.py

Training loop za BlendshapeGRU / BlendshapeTCN / BlendshapeTransformer.

    from scripts.train import train

    # MFCC mod
    train(
        model_type  = "gru",        # "gru" | "tcn" | "transformer"
        audio_type  = "mfcc",       # "mfcc" | "hubert"
        data_root   = "data",
        epochs      = 50,
        batch_size  = 4,
        lr          = 1e-3,
        device      = "cuda",
    )

    # HuBERT mod
    train(
        model_type  = "tcn",
        audio_type  = "hubert",
        hubert_dir  = "/content/hubert_features",
        data_root   = "data",
        epochs      = 50,
        device      = "cuda",
    )
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from torch.utils.data import DataLoader, random_split

from src.config import N_BLENDSHAPES
from src.models.gru         import BlendshapeGRU
from src.models.tcn         import BlendshapeTCN
from src.models.transformer import BlendshapeTransformer
from src.models.losses      import combined_loss
from src.utils.dataset      import BlendshapeDataset, collate_fn_mfcc, collate_fn_hubert
from src.utils.Visualization.results_manager import ResultsManager



def _build_model(model_type: str, audio_type: str, device: str, **kwargs):
    # inicijalizuje odgovarajuci model
    common = dict(audio_type=audio_type)
    common.update(kwargs)

    if model_type == "gru":
        model = BlendshapeGRU(**common)
    elif model_type == "tcn":
        model = BlendshapeTCN(**common)
    elif model_type == "transformer":
        # transformer nema audio_type / use_phonemes parametar u trenutnoj verziji - fixovano
        # common.pop("audio_type", None)
        # common.pop("use_phonemes", None)
        model = BlendshapeTransformer(**common)
    else:
        raise ValueError(f"Nepoznat model_type: {model_type}. Koristiti: gru | tcn | transformer")

    print(f"[Train] Model: {model_type.upper()}  audio_type={audio_type}")
    print(f"[Train] Parametara: {model.count_params():,}")
    return model.to(device)


def _run_epoch(model, loader, optimizer, device, audio_type, is_train,
               vel_lam=0.5, acc_lam=0.1):
    # jedna epoha - train/val
    model.train() if is_train else model.eval()

    total_mse = total_vel = total_acc = total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    phase = "train" if is_train else "val"
    with ctx:
        for batch in tqdm(loader, desc=phase, leave=False):
            af      = batch["audio_feats"].to(device)   # (B, T, FEAT_DIM)
            pi      = batch["phoneme_ids"].to(device)    # (B, T)
            pt      = batch["phoneme_trel"].to(device)   # (B, T, 1)
            si      = batch["speaker_ids"].to(device)    # (B, T)
            targets = batch["targets"].to(device)        # (B, T, 52)
            lengths = batch["lengths"].to(device)        # (B,)
            mask    = batch["mask"].to(device)           # (B, T)

            hubert = None
            if audio_type == "hubert":
                hubert = batch["hubert_feats"].to(device)  # (B, T, 768)

            pred = model(af, pi, pt, si, lengths=lengths, hubert=hubert)  # (B, T, 52)

            loss, components = combined_loss(pred, targets, mask, vel_lam=vel_lam, acc_lam=acc_lam)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_mse  += components["mse"]
            total_vel  += components["vel"]
            total_acc  += components["acc"]
            n_batches  += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "mse":  total_mse  / n,
        "vel":  total_vel  / n,
        "acc":  total_acc  / n,
    }

# main 

def train(
    # Podaci
    data_root:    str   = "data",
    speakers:     list  = ["spk08", "spk14"],
    hubert_dir:   str   = None,
    val_split:    float = 0.15,
    load_synth:   bool  = False,
    augment:      bool  = True,

    # Model
    model_type:   str   = "gru",       # "gru" | "tcn" | "transformer"
    audio_type:   str   = "mfcc",      # "mfcc" | "hubert"
    d_model:      int   = 256,
    use_phonemes: bool  = True,

    # Trening
    epochs:       int   = 50,
    batch_size:   int   = 4,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int   = 10,          # early stopping
    device:       str   = "cuda",

    # Cuvanje
    results_root: str   = "/content/results",
    ckpt_every:   int   = 5,           # cuva checkpoint svakih N epoha
    display_inline: bool = True,

    # Tezine gubitka
    vel_lam:      float = 0.5,
    acc_lam:      float = 0.1,
) -> str:
    """
    Trenira model i cuva rezultate u /content/results/<model>_<timestamp>/.

    Returns
    -------
    str : putanja do session foldera
    """

    device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("[Train] CUDA nije dostupna — koristim CPU")
        device = "cpu"
    print(f"[Train] Device: {device}")

    print(f"[Train] Data root  : {data_root}")
    print(f"[Train] Speakers   : {speakers}")

    # Dataset
    ds = BlendshapeDataset(
        data_root        = data_root,
        speakers         = speakers,
        augment          = False,       # augment samo na train splitu
        load_synth       = load_synth,
        use_preprocessing= True,
        hubert_dir       = hubert_dir,
    )

    if len(ds) == 0:
        raise RuntimeError(
            f"[Train] Dataset is empty. "
            f"Check that data exists at: {data_root}\n"
            f"Expected structure: {data_root}/spk08_blendshapes/renamed_spk08/*.csv + *.wav"
        )

    n_val   = max(1, int(len(ds) * val_split))
    n_train = max(1, len(ds) - n_val)
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Augmentacija samo na train
    train_ds.dataset.augment = augment

    collate = collate_fn_hubert if audio_type == "hubert" else collate_fn_mfcc

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0, pin_memory=True)

    print(f"[Train] Train: {n_train}  Val: {n_val}  Batch: {batch_size}")

    # Model 
    model = _build_model(
        model_type   = model_type,
        audio_type   = audio_type,
        device       = device,
        d_model      = d_model,
        use_phonemes = use_phonemes,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    ) 
    # scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    # ResultsManager 
    rm = ResultsManager(
        model_name    = f"{model_type}_{audio_type}",
        results_root  = results_root,
        display_inline= display_inline,
    )
    ckpt_dir = rm.ckpt_dir

    rm.save_config({
        "model_type":   model_type,
        "audio_type":   audio_type,
        "d_model":      d_model,
        "use_phonemes": use_phonemes,
        "epochs":       epochs,
        "batch_size":   batch_size,
        "lr":           lr,
        "weight_decay": weight_decay,
        "patience":     patience,
        "val_split":    val_split,
        "augment":      augment,
        "speakers":     speakers,
        "n_train":      n_train,
        "n_val":        n_val,
        "n_params":     model.count_params(),
    })

    # Training loop 
    best_val_loss  = float("inf")
    best_ckpt_path = None
    patience_count = 0

    for epoch in range(1, epochs + 1):

        train_metrics = _run_epoch(model, train_loader, optimizer, device, audio_type, is_train=True,  vel_lam=vel_lam, acc_lam=acc_lam)
        val_metrics   = _run_epoch(model, val_loader,   optimizer, device, audio_type, is_train=False, vel_lam=vel_lam, acc_lam=acc_lam)

        scheduler.step(val_metrics["loss"])
        rm.log_epoch(epoch, train=train_metrics, val=val_metrics)

        print(
            f"Epoha {epoch:3d}/{epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_mse={val_metrics['mse']:.4f}"
        )

        # Checkpoint svake N epoha
        if epoch % ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_metrics["loss"],
            }, ckpt_path)
            rm.register_checkpoint(ckpt_path)

        # Best model checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss  = val_metrics["loss"]
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_metrics["loss"],
                "config": {
                    "model_type":   model_type,
                    "audio_type":   audio_type,
                    "d_model":      d_model,
                    "use_phonemes": use_phonemes,
                    "speakers":     speakers,
                    "vel_lam":      vel_lam,
                    "acc_lam":      acc_lam,
                },
            }, best_ckpt_path)
            patience_count = 0
        else:
            patience_count += 1

        # Early stopping
        if patience_count >= patience:
            print(f"[Train] Early stopping na epohi {epoch} — val_loss nije poboljsan {patience} epoha.")
            break

    # Grafici i summary 
    rm.save_loss_curves()

    # Vizualizacija na prvom val uzorku
    try:
        sample_idx = val_ds.indices[0]
        sample     = ds[sample_idx]
        T = sample["length"]

        model.eval()
        with torch.no_grad():
            af = sample["audio_feats"].unsqueeze(0).to(device)
            pi = sample["phoneme_ids"].unsqueeze(0).to(device)
            pt = sample["phoneme_trel"].unsqueeze(0).to(device)
            si = sample["speaker_ids"].unsqueeze(0).to(device)
            hb = sample.get("hubert_feats")
            hb = hb.unsqueeze(0).to(device) if hb is not None else None

            pred_np   = model(af, pi, pt, si, lengths=None, hubert=hb).squeeze(0).cpu().numpy()
            target_np = sample["targets"].numpy()
            mfcc_np   = sample["audio_feats"].numpy()
            ph_ids_np = sample["phoneme_ids"].numpy()
            ph_rel_np = sample["phoneme_trel"].squeeze(-1).numpy()
            mse_bs    = np.mean((pred_np - target_np) ** 2, axis=0)

        rm.save_per_blendshape_mse(mse_bs)
        rm.save_error_correlation(pred_np, target_np)
        rm.save_prediction(
            pred_np, target_np,
            mfcc_feats   = mfcc_np,
            phoneme_ids  = ph_ids_np,
            phoneme_trel = ph_rel_np,
            name         = sample["name"],
            save_all     = True,
        )
    except Exception as e:
        print(f"[Train] Vizualizacija preskocena: {e}")

    rm.save_summary({
        "best_val_loss":  best_val_loss,
        "best_ckpt":      best_ckpt_path,
        "total_epochs":   epoch,
        "model_type":     model_type,
        "audio_type":     audio_type,
        "n_params":       model.count_params(),
    })

    session_path = rm.finalize()
    print(f"\n[Train] Best val_loss : {best_val_loss:.4f}")
    print(f"[Train] Best ckpt     : {best_ckpt_path}")
    print(f"[Train] Rezultati     : {session_path}")

    return session_path


def main():
    parser = argparse.ArgumentParser(description="Train blendshape prediction model.")

    # Model
    parser.add_argument("--model",        type=str,   default="gru",
                        choices=["gru", "tcn", "transformer"])
    parser.add_argument("--audio",        type=str,   default="mfcc",
                        choices=["mfcc", "hubert"])
    parser.add_argument("--d_model",      type=int,   default=256)
    parser.add_argument("--no_phonemes",  action="store_true", default=False,
                        help="Disable phoneme features")

    # Data
    parser.add_argument("--data_root",    type=str,   default="data")
    parser.add_argument("--results_root", type=str,   default="/content/results")
    parser.add_argument("--hubert_dir",   type=str,   default=None)
    parser.add_argument("--speakers",     type=str,   nargs="+", default=["spk08", "spk14"])
    parser.add_argument("--use_synthetic",action="store_true", default=False)
    parser.add_argument("--no_augment",   action="store_true", default=False,
                        help="Disable data augmentation")

    # Training
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=4)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--patience",         type=int,   default=10)
    parser.add_argument("--checkpoint_every", type=int,   default=5)
    parser.add_argument("--device",           type=str,   default="cuda")

    # Loss weights
    parser.add_argument("--vel_lam", type=float, default=0.5)
    parser.add_argument("--acc_lam", type=float, default=0.1)

    args = parser.parse_args()

    train(
        model_type     = args.model,
        audio_type     = args.audio,
        d_model        = args.d_model,
        use_phonemes   = not args.no_phonemes,
        data_root      = args.data_root,
        results_root   = args.results_root,
        hubert_dir     = args.hubert_dir,
        speakers       = args.speakers,
        load_synth     = args.use_synthetic,
        augment        = not args.no_augment,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        patience       = args.patience,
        ckpt_every     = args.checkpoint_every,
        device         = args.device,
        vel_lam        = args.vel_lam,
        acc_lam        = args.acc_lam,
        display_inline = False,
    )


if __name__ == "__main__":
    main()