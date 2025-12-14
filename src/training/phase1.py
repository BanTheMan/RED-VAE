import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from src.training.metrics import ensure_dir, save_loss_plot, append_metrics_csv

# Import your modules
from src.inference.generator import ARCGenerator
from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.LViTM.body import LargeVisionTransformerModel
# from src.architecture.executor.diffuser import Executor
from src.architecture.ViT.body import VisionTransformer

# Training constants
LR = 1e-4
EPOCHS = 75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11


##############################################################
#   Build MODEL for PHASE 1 using TWO separate ViTs
##############################################################
def build_model():

    img_size   = 30
    patch_size = 1
    embed_dim  = 256
    num_heads  = 4
    depth_vit  = 6
    mlp_dim    = 512
    z_dim      = 64
    num_props  = 4

    # For (I,O) pairs
    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2
    ).to(DEVICE)

    # For test input
    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1
    ).to(DEVICE)

    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)
    aggregator = ExamplePairAggregator(embed_dim=vit_pair.c_token.size(-1)).to(DEVICE)
    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=8,
        mlp_dim=256,
        depth=10,
        num_proposals=num_props,
        z_dim=z_dim
    ).to(DEVICE)

    executor = Executor(
        embed_dim=vit_pair.c_token.size(-1),
        num_heads=4,
        mlp_dim=512,
        depth=6,
        z_dim=z_dim,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    generator = ARCGenerator(
        example_pair_encoder=example_encoder,
        aggregator=aggregator,
        cond_encoder=cond_encoder,
        lvitm=lvitm,
        executor=executor
    ).to(DEVICE)

    return generator


##############################################################
#   PHASE 1 TRAINING LOOP
##############################################################
def train_phase1(arc_loader):

    generator = build_model()
    optimizer = Adam(generator.parameters(), lr=LR)
    generator.train()
    # Ensure checkpoint directory exists and track train losses
    ensure_dir("checkpoints")
    train_losses = []

    for epoch in tqdm(range(EPOCHS), desc="Epoch:"):
        # print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        total_loss = 0.0
        count = 0

        for batch in tqdm(arc_loader, desc="Batch:"):

            ############################################################
            #   Move batch to device
            ############################################################
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            # === FIX 1: All masks must be boolean ===
            batch["train_input_masks"]  = batch["train_input_masks"].bool()
            batch["train_output_masks"] = batch["train_output_masks"].bool()
            batch["test_input_masks"]   = batch["test_input_masks"].bool()
            batch["test_output_masks"]  = batch["test_output_masks"].bool()

            # Extract tensors
            train_inputs       = batch["train_inputs"]
            train_outputs      = batch["train_outputs"]
            train_input_masks  = batch["train_input_masks"]
            train_output_masks = batch["train_output_masks"]

            test_inputs        = batch["test_inputs"]
            test_outputs       = batch["test_outputs"]
            test_input_masks   = batch["test_input_masks"]
            test_output_masks  = batch["test_output_masks"]

            ############################################################
            #   Forward through ARCGenerator
            ############################################################
            out = generator(
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                train_input_masks=train_input_masks,
                train_output_masks=train_output_masks,
                test_inputs=test_inputs,
                test_input_masks=test_input_masks,
            )

            logits = out["logits"]  # (B, K_test, C_out, H, W)

            # === FIX 3: sanitize logits to prevent NaN/Inf propagation ===
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            B, K_test, C_out, H, W = logits.shape

            ############################################################
            #   Clean CE Loss (Masked)
            ############################################################
            logits_flat = logits.view(B * K_test, C_out, H, W)
            target_flat = test_outputs.view(B * K_test, H, W)

            PAD_TOKEN = -100
            targets = target_flat.clone()

            pad_mask = ~test_output_masks.view(B*K_test, H, W).bool()
            targets[pad_mask] = PAD_TOKEN

            per_pixel = F.cross_entropy(
                logits_flat,
                targets,
                ignore_index=PAD_TOKEN,
                reduction="none"
            )

            valid_mask = (targets != PAD_TOKEN).float()
            loss = (per_pixel * valid_mask).sum() / valid_mask.sum()

            ############################################################
            #   Backprop
            ############################################################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Append and save evolving loss plot (no display)
        train_losses.append(avg_loss)
        try:
            save_loss_plot(train_losses, None, out_path="checkpoints/phase1_loss.png")
            append_metrics_csv("checkpoints/phase1_metrics.csv", {"epoch": epoch + 1, "train_loss": avg_loss})
        except Exception as e:
            print(f"[phase1] Warning: failed to save loss plot/metrics: {e}")

    return generator


##############################################################
#   Script Entry Point
##############################################################
# if __name__ == "__main__":
#     from src.data_pipeline.dataloader import ARCDataModule
#
#     model = train_phase1(ARCDataModule)
#     torch.save(model.state_dict(), "phase1_generator.pt")
#     print("Saved Phase 1 generator.")
