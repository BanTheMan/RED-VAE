import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm

from src.training.ppo_actor import PPOActor
from src.training.ppo_value import PPOValuer
from src.training.ppo_refiner import PPORefiner

from src.architecture.context_encoding.example_pair_encoder import ExamplePairEncoder
from src.architecture.context_encoding.example_pair_aggregator import ExamplePairAggregator
from src.architecture.context_encoding.conditional_encoder import ConditionalTestInputEncoder
from src.architecture.ViT.body import VisionTransformer
from src.architecture.LViTM.body import LargeVisionTransformerModel
from src.architecture.executor.executor import Executor
from src.architecture.adViT.critic import AdversarialVisionTransformer

from src.data_pipeline.dataloader import ARCDataModule
from src.inference.execution_controller import HybridExecuteController
from src.training.metrics import ensure_dir, save_loss_plot, append_metrics_csv


############################################################
#              ** PPO HYPERPARAMETERS **
############################################################
PPO_EPOCHS = 25
PPO_STEPS = 10          # PPO rollout length
PPO_GAMMA = 0.99       # reward discount
PPO_LAMBDA = 0.95      # GAE lambda
PPO_CLIP = 0.2
PPO_LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 11
Z_DIM = 64
NUM_PROPOSALS = 4


############################################################
#            Build All Components Cleanly
############################################################
def build_generator_components():
    img_size = 30
    patch = 1
    embed_dim = 256
    heads = 4
    depth_vit = 6
    mlp_dim = 512

    vit_pair = VisionTransformer(
        img_size=img_size,
        patch_size=patch,
        embed_dim=embed_dim,
        num_heads=heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=2
    ).to(DEVICE)

    vit_test = VisionTransformer(
        img_size=img_size,
        patch_size=patch,
        embed_dim=embed_dim,
        num_heads=heads,
        depth=depth_vit,
        mlp_dim=mlp_dim,
        in_channels=1
    ).to(DEVICE)

    example_encoder = ExamplePairEncoder(vit_pair).to(DEVICE)
    aggregator = ExamplePairAggregator(embed_dim).to(DEVICE)
    cond_encoder = ConditionalTestInputEncoder(vit_test).to(DEVICE)

    lvitm = LargeVisionTransformerModel(
        embed_dim=embed_dim,
        num_heads=heads,
        mlp_dim=256,
        depth=8,
        num_proposals=NUM_PROPOSALS,
        z_dim=Z_DIM
    ).to(DEVICE)

    executor = Executor(
        embed_dim=embed_dim,
        num_heads=heads,
        mlp_dim=512,
        depth=6,
        z_dim=Z_DIM,
        hidden_channels=64,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    return example_encoder, aggregator, cond_encoder, lvitm, executor



def build_critic():
    vit = VisionTransformer(
        img_size=30,
        patch_size=1,
        embed_dim=256,
        num_heads=4,
        depth=4,
        mlp_dim=256,
        in_channels=2
    ).to(DEVICE)

    return AdversarialVisionTransformer(
        vit_encoder=vit,
        z_dim=None,
        c_dim=None,
        mlp_dim=256
    ).to(DEVICE)



############################################################
#         Compute Transformer Context C
############################################################
def compute_context(example_encoder, aggregator, train_inputs, train_outputs,
                    train_input_masks, train_output_masks):

    if train_inputs.dim() == 3:
        train_inputs = train_inputs.unsqueeze(0)
        train_outputs = train_outputs.unsqueeze(0)
        train_input_masks = train_input_masks.unsqueeze(0)
        train_output_masks = train_output_masks.unsqueeze(0)

    B, K, H, W = train_inputs.shape
    h_list = []

    for k in range(K):
        I_k = train_inputs[:, k].unsqueeze(1).float()
        O_k = train_outputs[:, k].unsqueeze(1).float()

        mI = train_input_masks[:, k]
        mO = train_output_masks[:, k]

        h_list.append(example_encoder(I_k, O_k, mI, mO))

    h = torch.stack(h_list, dim=1)
    return aggregator(h, mask=None)



############################################################
#              PPO Rollout + Update
############################################################
def ppo_rollout(controller, init_grid, init_mask, C, actor, valuer):
    """
    Run a PPO rollout:
       (1) sample z adjustments
       (2) calculate reward from critic
       (3) compute logprobs, advantages, returns
    """

    B, _, H, W = init_grid.shape

    # Storage
    states = []
    actions = []
    logprobs = []
    values = []
    rewards = []

    grid = init_grid.clone()

    for t in range(PPO_STEPS):

        # Encode for LVITM proposals
        tokens, padmask = controller.cond_encoder(grid, init_mask, C)
        Z = controller.lvitm(C, tokens, padmask)  # (B,T,z_dim)

        # Actor samples "action" = Δz shift
        mu, log_std = actor(Z)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        action = dist.sample()           # (B,T,z_dim)
        logp = dist.log_prob(action).sum(dim=-1)  # (B,T)

        # Apply action (shift proposals)
        Z_refined = Z + 0.1 * action

        # Execute each proposal
        B,T,z_dim = Z_refined.shape
        grid_batch = grid.unsqueeze(1).expand(B,T,1,H,W).reshape(B*T,1,H,W)
        z_flat = Z_refined.reshape(B*T,z_dim)

        logits = controller.executor(grid_batch, z_flat)
        logits = logits.view(B,T,NUM_CLASSES,H,W)

        # Critic gives reward
        reward = controller.critic(
            I_in=grid,
            O_pred=logits,
            mask_in=init_mask,
            mask_out=init_mask,
            z=Z_refined,
            C=C
        )  # (B,T)

        # Value estimate
        value = valuer(Z)  # (B,T)

        # Store trajectory
        states.append(Z)
        actions.append(action)
        logprobs.append(logp)
        values.append(value)
        rewards.append(reward)

        # Select best output for next state
        best_idx = reward.argmax(dim=1)  # (B)
        idx = best_idx.view(B,1,1,1,1).expand(B,1,NUM_CLASSES,H,W)
        best_logits = logits.gather(dim=1, index=idx).squeeze(1)
        grid = best_logits.argmax(dim=1, keepdim=True).float()

    # Convert lists to tensors:
    states = torch.stack(states, dim=0)      # (T,B,P,D)
    actions = torch.stack(actions, dim=0)    # (T,B,P,D)
    logprobs = torch.stack(logprobs, dim=0)  # (T,B,P)
    values = torch.stack(values, dim=0)      # (T,B,P)
    rewards = torch.stack(rewards, dim=0)    # (T,B,P)

    # Record mean reward for this rollout (T,B,P) -> scalar
    try:
        ensure_dir("checkpoints")
        mean_reward = float(rewards.mean().item())
        append_metrics_csv("checkpoints/phase4_ppo_rollouts.csv", {"mean_reward": mean_reward})
    except Exception:
        # Do not break rollout if metrics saving fails
        pass

    return states, actions, logprobs, values, rewards



def compute_gae(values, rewards):
    """
    Generalized Advantage Estimation (GAE-Lambda)
    """
    T, B, P = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, P, device=DEVICE)

    for t in reversed(range(T)):
        delta = rewards[t] + PPO_GAMMA * (values[t+1] if t < T-1 else 0) - values[t]
        gae = delta + PPO_GAMMA * PPO_LAMBDA * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns



############################################################
#                   ** PHASE 4 MAIN **
############################################################
def train_phase4_ppo(data_loader):
    # Build models
    example_encoder, aggregator, cond_encoder, lvitm, executor = build_generator_components()
    critic = build_critic()

    controller = HybridExecuteController(
        lvitm=lvitm,
        executor=executor,
        cond_encoder=cond_encoder,
        critic=critic
    ).to(DEVICE)

    actor = PPOActor(z_dim=Z_DIM, embed_dim=256).to(DEVICE)
    valuer = PPOValuer(z_dim=Z_DIM, embed_dim=256).to(DEVICE)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(valuer.parameters()), lr=PPO_LR)

    # MAIN TRAIN LOOP
    for epoch in tqdm(range(PPO_EPOCHS), "PPO Epoch:"):
        # print(f"\n==== PPO Epoch {epoch+1}/{PPO_EPOCHS} ====")

        for batch_idx, batch in enumerate(data_loader):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(DEVICE)

            train_in = batch["train_inputs"]
            train_out = batch["train_outputs"]
            train_inm = batch["train_input_masks"]
            train_outm = batch["train_output_masks"]

            test_in = batch["test_inputs"]
            test_inm = batch["test_input_masks"]
            test_out = batch["test_outputs"]

            C = compute_context(example_encoder, aggregator,
                                train_in, train_out, train_inm, train_outm)

            if test_in.dim() == 3:
                test_in = test_in.unsqueeze(0)
                test_inm = test_inm.unsqueeze(0)
                test_out = test_out.unsqueeze(0)

            B, K, H, W = test_in.shape
            init_grid = test_in[:,0].unsqueeze(1).float()
            init_mask = test_inm[:,0]

            # === PPO Rollout ===
            states, actions, old_logp, values, rewards = ppo_rollout(
                controller, init_grid, init_mask, C, actor, valuer
            )

            advantages, returns = compute_gae(values, rewards)

            # === PPO UPDATE ===
            optimizer.zero_grad()

            T,B,P,D = states.shape
            states = states.detach()
            actions = actions.detach()
            advantages = advantages.detach()
            returns = returns.detach()
            old_logp = old_logp.detach()

            mu, log_std = actor(states)
            dist = Normal(mu, torch.exp(log_std))
            new_logp = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_logp - old_logp)
            clipped = torch.clamp(ratio, 1-PPO_CLIP, 1+PPO_CLIP)

            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = F.mse_loss(valuer(states), returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5*value_loss - 0.01*entropy
            loss.backward()
            optimizer.step()

            print(f"[PPO] Batch {batch_idx} | loss={loss.item():.4f} | policy={policy_loss.item():.4f}")

    return actor, valuer



if __name__ == "__main__":
    train_phase4_ppo(ARCDataModule)
