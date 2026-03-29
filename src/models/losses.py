import torch
from src.config import N_BLENDSHAPES, BLENDSHAPE_NAMES, MOUTH_INDICES

def build_weights(device, mouth_w=5.0, jaw_tongue_w=4.0):
    w = torch.ones(N_BLENDSHAPES, device=device)
    for i, n in enumerate(BLENDSHAPE_NAMES):
        if 'jaw' in n or 'tongue' in n: w[i] = jaw_tongue_w
        elif 'mouth' in n:              w[i] = mouth_w
    return w

def combined_loss(pred, target, mask, mouth_w=5.0, jaw_w=4.0,
                  vel_lam=2.0, acc_lam=0.5):
    w = build_weights(pred.device, mouth_w, jaw_w) # shape (52, )
    # mask - True (real) False (synth)
    L_mse = (((pred - target)**2) * w)[mask].mean()

    # velocity - how much does the blendshape change between consecutive frames
    pv    = pred[:,1:,:]   - pred[:,:-1,:]
    # same for targets
    gv    = target[:,1:,:] - target[:,:-1,:]
    mv    = mask[:,1:] & mask[:,:-1]
    L_vel = (((pv - gv)**2) * w)[mv].mean()
    # acceleration - rate of change of velocity
    pa    = pv[:,1:,:]  - pv[:,:-1,:]
    ga    = gv[:,1:,:]  - gv[:,:-1,:]
    ma    = mv[:,1:] & mv[:,:-1]
    L_acc = (((pa - ga)**2) * w)[ma].mean()

    return L_mse + vel_lam*L_vel + acc_lam*L_acc, \
           dict(mse=L_mse.item(), vel=L_vel.item(), acc=L_acc.item())

def weighted_mse_loss(pred, target, mask, mouth_weight=3.0):
    """Simple weighted MSE, kept for backwards compatibility."""
    w = torch.ones(N_BLENDSHAPES, device=pred.device)
    w[MOUTH_INDICES] = mouth_weight  # noqa — needs MOUTH_INDICES import
    diff = ((pred - target)**2) * w
    return diff[mask].mean()