import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from . import autograd_hacks


def deactivate_batchnorm(m):
    """
    Decativates BatchNorm layers during the NTK matrix computation.
    """
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


def compute_ntk_score_batched_permute(net, inputs, targets, mode, n_batch=1, n_permute=5, use_hack=True, diagonal_I_mag=1e-4, get_eigen=False, return_raw=False, debug=False):
    """
    Computes the NTK (in-domain generalization) score using the diagonal blocks approximation with permutations.

    Args:
        n_permute (int): the number of permutations to take average from
        other args refer to compute_ntk_score_batched fn

    Return:
        the NTK in-domain generalization score
    """
    scores_sum = 0
    min_eigen = None
    N = inputs.shape[0]
    for perm_i in range(n_permute):
        if debug:
            print(f"  [NTK DBG] permute {perm_i+1}/{n_permute}")
        indices = torch.randperm(N)
        score, permute_min_eigen = compute_ntk_score_batched(
            net, inputs[indices], targets[indices], mode,
            n_batch=n_batch, use_hack=use_hack, diagonal_I_mag=diagonal_I_mag,
            get_eigen=get_eigen, return_raw=return_raw, debug=debug)
        if debug:
            print(f"  [NTK DBG] permute {perm_i+1} score={score:.6f}")
        scores_sum += score
        if get_eigen:
            min_eigen = permute_min_eigen if min_eigen == None else np.min([min_eigen, permute_min_eigen])
    avg = scores_sum / n_permute
    if debug:
        print(f"  [NTK DBG] average over {n_permute} permute(s) = {avg:.6f}")
    return avg, min_eigen


def compute_ntk_score_batched(net, inputs, targets, mode, n_batch, use_hack, diagonal_I_mag, get_eigen=False, return_raw=False, debug=False):
    """
    Block diagonal matrix construction of NTK.
    The computation is simplified due to the inverse and multiplication of blocked diagonal matrices.

    Args:
        net (torch nn): the neural network
        inputs (Tensor): inputs of the training set
        targets (Tensor): labels of the training set
        mode (str): 'cls' for classfication; 'reg' for regression
        n_batch (int): the number of diagonal blocks
        use_hack (bool): whether to use the autograd_hack trick for per-example gradient
        diagonal_I_mag (float): the magnitude of the diagonal identity matrix
        get_eigen (bool): whether to output the eigenvalue of the NTK matrix (for robustness to model experiment only)
        return_raw (bool): if True, return (y theta^{-1} y); if False, return the whole term under sqrt
        debug (bool): if True, print intermediate shapes and statistics

    Return:
        the score, eigenvalue (which is None if get_eigen==True)
    """

    def dbg(msg):
        if debug:
            print(f"  [NTK DBG] {msg}")

    def loss_fn(preds, targets, mode='cls'):
        # Loss function for NTK computation (i.e., the model output)
        N = preds.shape[0]
        if mode == 'cls':
            return sum(preds[torch.arange(N), targets])
        else:
            return sum(preds)

    def get_weights(net):
        # Get all applicable weights
        weights = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weights.append(layer.weight)
                layer.weight.requires_grad_(True)
        return weights

    net.apply(deactivate_batchnorm)
    net.zero_grad()
    weights = get_weights(net)
    N = inputs.shape[0]
    score = 0
    min_eigen = None

    dbg(f"inputs shape={tuple(inputs.shape)}  targets shape={tuple(targets.shape)}  "
        f"mode={mode}  n_batch={n_batch}  diag_I={diagonal_I_mag}")
    if debug and mode == 'reg':
        dbg(f"targets stats — mean={targets.float().mean():.4f}  "
            f"std={targets.float().std():.4f}  "
            f"min={targets.float().min():.4f}  max={targets.float().max():.4f}")

    for sp in range(n_batch):
        # For each block matrix, calculate the NTK score
        st = sp * N // n_batch
        en = (sp + 1) * N // n_batch

        if st == en:
            # Avoid errors when N < n_batch
            continue

        dbg(f"--- block {sp+1}/{n_batch}  samples [{st}:{en}]  block_size={en-st} ---")

        grads = []
        fx = []
        net.zero_grad()
        net.apply(deactivate_batchnorm)

        if use_hack:
            # Calculate the per-example gradients using the autograd_hack
            # Clear past backprop data
            autograd_hacks.clear_backprops(net)
            # Add hook only when not already added
            if not hasattr(net, 'autograd_hacks_hooks') or len(net.autograd_hacks_hooks)==0:
                autograd_hacks.add_hooks(net)
            outputs = net.forward(inputs[st:en])
            dbg(f"outputs shape={tuple(outputs.shape)}  "
                f"mean={outputs.detach().float().mean():.4f}  "
                f"std={outputs.detach().float().std():.4f}")
            loss_fn(outputs, targets[st:en], mode=mode).backward()
            autograd_hacks.compute_grad1(net, loss_type='sum')

            for param in net.parameters():
                # Only support Linear and Conv2D for now
                if hasattr(param, 'grad1'):
                    grads.append(param.grad1.flatten(start_dim=1))

            grads = torch.cat(grads, axis=1)
            dbg(f"grads (Jacobian) shape={tuple(grads.shape)}  "
                f"norm_mean={grads.norm(dim=1).mean():.4f}  "
                f"norm_std={grads.norm(dim=1).std():.4f}")
            if mode == 'cls':
                fx = outputs[np.arange(en-st), targets[st:en]]
            else:
                fx = outputs[np.arange(en-st)]
                dbg(f"fx (model preds) shape={tuple(fx.shape)}  "
                    f"mean={fx.detach().float().mean():.4f}  "
                    f"std={fx.detach().float().std():.4f}")
        else:
            # Calculate the per-example gradients sample wise
            weights = get_weights(net)
            for i in range(st, en):
                outputs = net.forward(inputs[i:i+1])
                if mode == 'cls':
                    fx += [outputs[0, targets[i]]]
                else:
                    fx += [outputs[0]]
                loss = loss_fn(outputs, targets[i], mode=mode)
                grad_w_p = autograd.grad(loss, weights, allow_unused=True)
                grad_w = torch.cat([g.reshape(-1) for g in grad_w_p], -1)
                grads += [grad_w]
                net.zero_grad()
                weights = get_weights(net)
            grads = torch.stack(grads, 0)
            fx = torch.stack(fx, 0)
            dbg(f"grads shape={tuple(grads.shape)} (sample-wise)  "
                f"norm_mean={grads.norm(dim=1).mean():.4f}")

        # Compute Y_hat
        if mode == 'cls':
            Y = torch.ones(en - st, device=grads.device) - fx
        else:
            Y = targets[st:en].reshape(-1, 1) - fx

        dbg(f"Y (residuals) shape={tuple(Y.shape)}  "
            f"mean={Y.detach().float().mean():.4f}  "
            f"std={Y.detach().float().std():.4f}  "
            f"||Y||²={Y.detach().float().pow(2).sum():.4f}")

        # Compute the score based on Y_hat and the NTK matrix
        H = torch.matmul(grads, grads.t())
        H += torch.diag(torch.zeros(grads.shape[0], device=grads.device) + diagonal_I_mag) # Avioid singular matrix
        eigenvalues_h, _ = torch.linalg.eigh(H)
        dbg(f"H shape={tuple(H.shape)}  "
            f"eig_min={eigenvalues_h[0].item():.4e}  "
            f"eig_max={eigenvalues_h[-1].item():.4e}  "
            f"cond={eigenvalues_h[-1].item() / (eigenvalues_h[0].item() + 1e-30):.4e}")
        if get_eigen:
            min_eigen = eigenvalues_h[0] if min_eigen == None else torch.min(min_eigen, eigenvalues_h[0])
        Hinv = torch.linalg.inv(H)
        Hinv_Y = torch.matmul(Hinv, Y.view(-1, 1))
        curr_score = torch.matmul(Y.view(1, -1), Hinv_Y)
        dbg(f"Yᵀ H⁻¹ Y = {curr_score.item():.6f}  "
            f"(contribution to score²×N = {curr_score.item():.6f})")
        score += curr_score

    final_score = torch.sqrt(score / N).item()
    dbg(f"==> final NTK score = √(Yᵀ H⁻¹ Y / N) = {final_score:.6f}  "
        f"(raw Yᵀ H⁻¹ Y = {score.item():.6f}  N={N})")

    if return_raw:
        return score, min_eigen.item() if min_eigen!= None else None
    else:
        return final_score, min_eigen.item() if min_eigen!= None else None