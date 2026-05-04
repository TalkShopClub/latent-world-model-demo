# Latent World Model for GUI Agents — Internal Demo Report

*2026-05-04 — combines project description, demo design, and two
conceptual videos.*

We are training a task-conditioned JEPA encoder + latent next-step
predictor on Ubuntu GUI trajectories. This report combines the project
context, the demo we are building to communicate it, and two conceptual
videos that anchor the story: one explains *how the imagined frames and
their attribution heatmaps are produced*, the other shows *why the
world model is useful*.

> The HTML version of this report (`index.html` in this directory)
> embeds the videos inline and is the recommended way to view it for an
> internal demo session.

---

## 1. Project description

We use a JEPA-based method to train a state encoder for GUI-related
tasks. Given the rich information present in raw screenshots, our
encoder is conditioned on the task. The goal is to learn representations
that capture only the features relevant to completing that task, rather
than encoding all visual details indiscriminately.

To achieve this in a self-supervised manner, we mask a large portion of
video frames and train the model to predict the missing content in
latent space, conditioned on the unmasked regions. The target latent
representations are themselves generated from the partially observed
(unmasked) inputs, ensuring consistency in the learning signal.

Through this JEPA-style training process, the model is encouraged to
focus on features that most strongly influence the trajectory of the
task. In practice, this leads to representations that emphasize
task-relevant structure while ignoring irrelevant visual noise. On top
of this encoder we train a next-step latent predictor, yielding a
latent-space world model capable of modeling task dynamics.

---

## 2. What the demo needs to show

A progress-bar demo conditioned on state and task is unconvincing — our
rollouts are not precise enough to support accurate progress estimation,
and a numeric gauge invites questions about calibration that do not
serve the marketing goal. The real strength of our approach is in two
capabilities, both of which the demo must make visceral:

- **Directional guidance.** The model gives agents a sense of
  *where to go*, not just how far along they are.
- **Outcome-based evaluation via rollouts.** By simulating candidate
  trajectories in latent space, we can compare different possible
  end-states and measure how close each lands to a desired goal state.
  The agent picks the action sequence whose imagined endpoint is
  closest to the goal.

The demo should make the agent's planning process feel tangible — like
watching a human mentally simulate options before committing.

---

## 3. Video 1 — How the imagined frames and attention heatmaps are made

**File:** `videos/latent_rollout_fig3_style.mp4`

A single 8-step rollout from a starting state `z_0` (encoded from a
real Desktop screenshot). Each step shows two panels side-by-side: the
*imagined frame* reconstructed from the latent, and the *attention
map* showing which regions the latent actually represents. The
per-step caption names the controlled object — "context menu",
"name field", "*R* typed into the name field", "OK as the commit
control", "*folder* created on the Desktop".

> White indicates attended regions; the underlined term names the
> controlled object.

The dressing in the imagined frame is the decoder filling in plausible
context — we are not selling generative fidelity. What we *are*
selling is that the highlighted regions in the attention map are
attributable to the latent and track the things the model is actually
reasoning about at each step.

### How the imagined frames are produced

1. **Encode the start.** Feed a real screenshot `x_0` and the task `T`
   into the frozen task-conditioned JEPA encoder:
   `z_0 = E(x_0 | T)`. The encoder is ViT-style, so `z_0` is a
   *sequence of patch tokens*, not a single global vector. This
   token-sequence form is important — it is what makes per-token
   attribution possible later.
2. **Roll forward in latent space.** Apply the next-step predictor
   step by step under a chosen action sequence:
   `z_{t+1} = P(z_t, a_t)`. The real GUI is never touched past step 0;
   everything past `x_0` is imagined.
3. **Decode each step to pixels.** A cross-attention diffusion decoder
   `D(z_t, T) → x̂_t` reconstructs the imagined screenshot at each
   step. The decoder cross-attends to `z_t`'s patch tokens, the same
   way Stable Diffusion cross-attends to text tokens.

### Why we need the attribution heatmap

A diffusion decoder fills in plenty of detail that does not come from
the latent — it comes from the prior. If we just showed the decoded
image, a viewer could not tell which pixels are the latent talking and
which are the decoder hallucinating. The heatmap closes that gap and,
more importantly, **scopes our claims**: we point at the bright region
and say "this is what the latent represents," and we honestly mark
everything else as decoder context.

### How the heatmap is computed

Three methods, used together, all enabled by the cross-attention decoder:

**(a) DAAM-style per-token attribution — primary.** For each latent
token, aggregate its cross-attention weights across all decoder
layers, attention heads, and diffusion timesteps. The result is a
per-token spatial heatmap showing where in the image that token
influenced pixels. Sum over all tokens to get the overall mask `m_t`
shown in the video. Same lineage as Prompt-to-Prompt / DAAM in
image-editing literature.

**(b) Variance-ratio map — sanity check.** Decode many times from
`z_t` with different diffusion noise to get per-pixel variance
`V_prior` (what the prior fills in freely). Decode from slightly
perturbed `z_t + ε` with the same noise schedule to get `V_z` (what
changes when the latent changes). The ratio
`V_z / (V_z + V_prior)` is a per-pixel "fraction of variance
attributable to z," architecture-agnostic. We use it to confirm DAAM
is not a cross-attention artifact — the two should agree on the rough
region.

**(c) Token swap — counterfactual demonstration.** Encode a paired
state (e.g., a frame with the dialog open and one with it closed),
swap the latent tokens DAAM identifies as "dialog tokens," and decode
again — the dialog appears or disappears. This is a *demonstration of
control*, not just an attribution: we prove the highlighted region is
real by editing it.

### Design choices that make this possible

- **Cross-attention decoder over `z` tokens, not a global vector.**
  Without this, methods (a) and (c) are not available. AdaLN/FiLM-style
  global conditioning would still work with method (b) only, which is
  much weaker as evidence.
- **Classifier-free-guidance dropout on `z` during decoder training.**
  Roughly 10% of training samples drop `z` and replace with a null
  token sequence. Costs nothing to add and gives us a fast "conditional
  vs unconditional" attribution signal at inference if we want it.
- **Train the decoder on rolled-out predicted latents too, not just
  encoder outputs.** The decoder is trained with pairs `(x, E(x|T))`
  from the demo bank, augmented with short-rollout latents
  `P(P(...P(z_0, a_0)...))` paired with their corresponding actual
  frames. Without this augmentation, decoded rollouts degrade exactly
  at the steps we care about because predicted latents drift off the
  encoder's manifold.
- **Honest blur.** JEPA latents are trained to discard pixel detail,
  so reconstructions are blurry on textures and sharp-ish on UI
  structure (dialog vs no-dialog, menu vs no-menu, cursor presence).
  This asymmetry is itself evidence that the latent has compressed
  away what does not matter and kept what does. We narrate it as a
  feature.

---

## 4. Video 2 — Why the world model is useful

**File:** `videos/latent_rollout_goal_selection.mp4` *(headline)*

From the same starting state `z_0`, the model rolls out three
candidate action sequences in latent space:

- **A — goal plan.** right-click → New Folder → type "Reports" → Enter.
- **B — distractor.** open the `Home` folder; file manager opens.
- **C — wrong commit.** open the dialog and click Cancel.

The endpoint of each rollout is compared to the goal latent `z_goal`
(encoded from the goal screenshot, shown as a reference card on the
right). The closest endpoint wins:

- Rollout A → 0.94
- Rollout B → 0.38
- Rollout C → 0.21

Rollout A is outlined in green and selected as the plan to execute.

### Why this matters for CUA-style agents

This is the planning operation that a CUA-style agent gets for free
from a latent world model. Without it, an agent has to commit to an
action and find out what happens; with it, the agent imagines a
handful of plans, scores their imagined endpoints against a goal
state, and only then commits — the same pattern a human uses when
mentally simulating options.

- **Think before acting.** Several candidate plans are rolled out in
  latent space, never touching the real GUI. Latent rollouts are
  cheap: no rendering, no GUI events, no waiting for pages to load.
- **Compare on outcomes, not heuristics.** The agent does not need a
  reward model or a hand-tuned scoring function — it has a goal state,
  and it asks "which imagined endpoint is closest to that goal?"
  Direct, model-internal, and task-grounded.
- **Reject obviously wrong branches.** The distractor (file manager)
  and the wrong-commit (dialog cancelled) rollouts score low and are
  filtered out before any real action is taken. This is exactly the
  kind of mistake a reactive CUA agent makes repeatedly today.
- **Recover from off-track states.** The same operation works
  mid-trajectory: re-encode the current screen, re-roll candidates,
  re-rank against the goal. The agent can re-plan instead of being
  committed to a stale plan.
- **Composes with any policy.** Candidate action sequences can come
  from a policy network's top-K, a hand-authored macro library, or a
  search procedure — the world model is the evaluator. It augments
  existing CUA stacks rather than replacing them.

The two videos together complete the story: **Video 1** shows that
the imagined frames are real and what the latent represents inside
them is auditable; **Video 2** shows that this auditable
representation is enough to drive a useful planning loop.

---

## 5. What we are claiming, and what we are not

| **Claim** | **Not a claim** |
| --- | --- |
| Latent rollout produces specific, recognizable imagined futures. | Pixel-level fidelity of the decoded screenshots. |
| Highlighted regions are attributable to the latent (DAAM, variance-ratio cross-check, token-swap counterfactual). | Every region of the imagined frame is latent-controlled — the rest is decoder context, openly labeled. |
| Different action sequences produce different, semantically appropriate imagined futures. | Latent distance is a calibrated progress metric. We use endpoint-to-goal similarity for selection only, not for monotone progress. |
| Endpoints can be compared to a goal latent to select an action plan. | The demo currently runs against a live policy — v1 ships curated rollouts. |

---

## 6. Status and open decisions

- **Conceptual videos:** two produced and embedded above. Storyboard
  quality, not yet model output.
- **Demo task:** "Create a folder named Reports on the Desktop."
  Locked for v1.
- **Decoder:** not yet trained. Cross-attention architecture is the
  recommended commitment so DAAM and token-swap are available; CFG
  dropout on `z` is essentially free to add and unlocks method (b).
- **Attribution method:** DAAM as headline overlay, variance-ratio
  backstage, token-swap as a punchline cut.
- **Live agent:** stretch goal; v1 ships curated rollouts.
- **Win/Mac coverage:** out of scope for Phase 1; Ubuntu only.

---

*Source materials: `../README.md`, `../task.md`, `../draft.md`,
`../multi_trajectory_prompts.md`, `../concept_video/`.*
