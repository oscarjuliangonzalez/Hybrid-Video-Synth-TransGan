## Persona
You are an expert Data Scientist Engineer specialized in Biomedical Image Processing and Machine Learning methods for generation of synthetic images.

## Dev environment tips
- All ML learning tasks have to be implemented using Pytorch, an all supported operations should be implemented using mps (metal performance shaders) as device.
- Implement file processing functionalities using Open CV.
- All visualization and input file processing tools are and should always be in `video_process.py`.
- Ignore `README.md` when context is needed. Always prefer `copilot-instructions.md`.
- NEVER modify nor delete existing functions. Always create new ones.

## Testing instructions
- Always test any modifications to the base network in a new notebook. Never modify old notebooks as the objective is to keep track of the changes.
- Whenever testing involves a neural network (most of the case), output for training should always use tqdm progress bar, and testing should display progress of loss and other important parameters
so that the user can tell whether the model is working or iteration should be stopped.
- Information of the intended network is available on `p4.pdf`. You should always check available literature for the construction of the network, and it is available in `literature/`.

## PR instructions
- Title format: [<project_name>] <Title>

---

## Training Guidance: Fixing Mode Collapse in VQGAN (test.ipynb)

### Problem: Token Collapse
The original `test.ipynb` suffered from **codebook collapse**—the model converged to using only ~12 of 1024 available tokens within 2–3 epochs, limiting output diversity.

**Root causes:**
1. Weak VQ loss without explicit commitment penalties
2. No incentive for codebook diversity
3. Strong discriminator overwhelming generator early
4. BCE loss saturation causing gradient collapse

### Solution: Four-Part Fix

#### 1. **Entropy Regularizer** (Core anti-collapse mechanism)
Penalizes low entropy in token usage:
```python
soft_assign = softmax(-distance / temp)    # soft assignments
avg_probs = mean(soft_assign, dim=0)       # P(each token used)
entropy = -sum(avg_probs * log(avg_probs + eps))
entropy_loss = log(vocab_size) - entropy   # loss ∝ unused tokens
```
**Effect:** Gradient pushes encoder to spread usage across codebook. When 1024 tokens are used equally, entropy_loss ≈ 0 (no penalty). When collapsed to 12 tokens, entropy_loss is high → strong gradient to diversify.

#### 2. **Perplexity Monitoring**
Perplexity = exp(entropy) = "effective number of tokens in use"
- Perplexity = 1 → collapse (1 token)
- Perplexity = 50 → good diversity
- Perplexity = 1024 → ideal (all tokens used)

Watch `Perplexity` in progress bars: should rise from ~3 to 30+ as training proceeds.

#### 3. **Improved VQ Loss** (from VQ-VAE-2)
```python
codebook_loss = MSE(z_q, encoder_output.detach())   # codebook moves toward encoder
commitment_loss = MSE(encoder_output, z_q.detach()) # encoder commits to codebook
vq_loss = codebook_loss + 0.25 * commitment_loss
```
**Why:** Asymmetric weighting prevents encoder from "abandoning" codebook by drifting away.

#### 4. **GAN Warmup + Hinge Loss**
- Train autoencoder alone for first 10 epochs (no GAN)
- Then introduce discriminator with **hinge loss** (not BCE)
  ```python
  loss_real = mean(relu(1 - logits_real))    # more stable than BCE
  loss_fake = mean(relu(1 + logits_fake))    # margins prevent saturation
  ```
- Discriminator trains at 0.5× generator LR (prevent overpowering)
- Gradient clipping at norm 1.0

**Why:** Matches NIPS-2014 GAN paper guidance on avoiding "Helvetica scenario" (collapse due to strong discriminator). Hinge loss has gradients even when confident.

### New Combined Objective
```
L_total = L1_recon + LPIPS_perceptual + VQ_loss + entropy_loss + (GAN_after_warmup)

Where:
  - L1_recon: robust reconstruction (sharper gradients than MSE)
  - LPIPS: perceptual similarity (medical image quality)
  - VQ_loss: commitment + codebook movement (meaningful codes)
  - entropy_loss: token diversity (anti-collapse)
  - GAN: adversarial realism (delayed until epoch 11)
```

### Expected Behavior
**Epochs 1–10 (warmup):**
- Tokens: 4 → 10 → 20+
- Perplexity: 3 → 10 → 30+
- No discriminator loss

**Epochs 11+ (adversarial):**
- Tokens: 50–200 (not collapse to 12)
- Perplexity: 40–100+ (sustained diversity)
- Disc_Loss: ~0.8–0.95 (healthy competition)

### Tuning Knobs
| Issue | Lever | Change |
|-------|-------|--------|
| Tokens still low | `TOKEN_ENTROPY_WEIGHT` | 0.02 → 0.05 |
| Tokens oscillate | `TOKEN_ENTROPY_TEMP` | 1.0 → 2.0 |
| Discriminator too strong | `GAN_WARMUP_EPOCHS` | 10 → 15 |
| Blurry outputs | `PERCEPTUAL_LOSS_WEIGHT` | 1.0 → 2.0 |

### Key References
- **VQ-VAE-2** (Razavi et al., 2019): commitment loss, codebook tricks
- **NIPS 2014 GAN paper**: D/G synchronization, avoiding collapse
- **LPIPS** (Zhang et al., 2018): perceptual loss for medical images
- Your literature: mode collapse when D too strong; mitigation via warmup + hinge loss