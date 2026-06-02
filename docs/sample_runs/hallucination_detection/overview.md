# Executive Summary

The tournament converged on a mechanistically coherent picture: LLM hallucination is not a single discrete failure mode but a family of internal states characterized by **uncertain or conflicted factual representations** that nonetheless project confidently into the output distribution. Both surviving hypotheses implicate mid-to-late transformer layers—specifically the feed-forward key-value memory subnetworks—as the locus where hallucination-prone states become identifiable via sparse autoencoder (SAE) feature activations. The key insight is that SAEs' interpretable, sparse decomposition allows researchers to move beyond black-box probing: they can characterize *what* the model "believes" at each layer and *whether* that belief is well-grounded. This matters because it opens a path to principled inference-time intervention (activation steering, feature suppression, or distribution sharpening) that is causally grounded rather than post-hoc correlational. If either direction validates causality on established benchmarks, the result would be the first scalable, interpretable hallucination-mitigation mechanism operable without retraining.

---

# Main Research Directions

## Direction 1: Hallucination-Subspace Suppression via Targeted SAE Feature Steering

**The direction.** Identify a compact cluster of monosemantic SAE features in mid-to-late residual stream positions that collectively encode "confident generation without factual grounding," then causally suppress this cluster at inference time using activation steering.

**Why it's promising.**
`[H-fb0242ebb9fccae6]` argues that hallucination-prone states are **sparse and localizable**—not diffusely distributed across all layers—which is precisely the regime in which SAEs excel. The core mechanistic claim is that these features correspond not merely to output uncertainty but to a qualitatively distinct internal state: high-confidence generation decoupled from the model's own factual retrieval circuitry. This is testable and, if true, implies that targeted suppression would reduce hallucination without broad degradation of fluency or capability. The framing inherits credibility from the broader mechanistic interpretability literature showing that SAE features in residual stream positions often correspond to identifiable semantic roles.

**Open questions.**
- Do "hallucination features" actually form a geometrically distinct cluster in SAE feature space, or are they scattered across many weakly correlated features? If the latter, suppression may require impractically many simultaneous interventions.
- Is the "confident-without-grounding" state monosemantic (captured by a small number of features) or polysemantic? Polysemanticity would make targeted steering much harder.
- Does suppressing these features reduce hallucination at the cost of increased refusals or hedging (a confound on TruthfulQA in particular)?
- **Falsification:** If activation steering on the identified feature cluster produces no statistically significant change in hallucination rates on TruthfulQA/HaluEval relative to random-feature steering baselines, the direction is undermined.

**First experiment.**
1. Train or obtain a pre-trained SAE on residual stream activations at layers corresponding to 60–80% of total model depth for a 7B-scale model (e.g., Mistral-7B or LLaMA-3-8B).
2. Run the model on a paired dataset of hallucinated vs. factual completions (drawn from TruthfulQA or HaluEval), record SAE feature activation vectors for each token at the decision point.
3. Use a sparse logistic probe *on the SAE feature space* (not raw activations) to identify features that discriminate the two conditions; verify these are interpretable (e.g., inspect top activating examples).
4. Apply activation steering (subtract or scale the identified feature directions) to held-out prompts and report $\Delta$-accuracy on TruthfulQA MC1 vs. a matched random-feature steering baseline.
*Estimated timeline: 6–8 weeks with one GPU cluster.*

---

## Direction 2: Epistemic Superposition Monitoring and Cross-Layer Consistency Enforcement

**The direction.** Hallucination arises when multiple semantically incompatible factual SAE features are simultaneously co-active at factual-retrieval layers (layers ~40–70% of depth), producing measurably elevated Shannon entropy over the sparse activation distribution; enforcing progressive concentration toward the dominant feature reduces hallucination without requiring labeled data.

**Why it's promising.**
`[H-77724168529242d6]` offers a richer mechanistic story than simple feature suppression: it predicts a *signature pattern* (high-entropy co-activation of mutually exclusive factual features) that is detectable online during inference without any hallucination-specific training labels. This is a significant practical advantage. The hypothesis also ties naturally to the mechanistic interpretability literature on "knowledge conflicts" in feed-forward layers and provides a quantitative observable (SAE activation entropy) that can be logged and thresholded at inference time—making it amenable to real-time deployment. The cross-layer consistency framing is novel: rather than intervening once, it proposes a multi-step sharpening that mirrors how humans resolve uncertainty by "committing" to a belief over time.

**Open questions.**
- Is elevated SAE activation entropy specific to hallucination-prone tokens, or does it also appear during legitimate hedging, novel/creative generation, or topic transitions? Specificity is critical for avoiding over-intervention.
- The "mutual exclusivity" assumption—that co-activated features are semantically incompatible—requires empirical validation. SAE features may frequently co-activate without representing conflicting facts.
- Does enforcing consistency (concentrating toward the dominant feature) ever *amplify* hallucination by committing to a wrong but confidently-activated feature? This is a meaningful failure mode.
- How is "dominant competing feature" selected when entropy is high? The intervention rule needs to be fully specified to be testable.
- **Falsification:** If SAE activation entropy at factual-retrieval layers is not significantly higher for hallucinated tokens than for correctly-recalled tokens (after controlling for token frequency and topic novelty), the mechanistic signature claimed does not exist.

**First experiment.**
1. Using the same 7B-scale model and a pre-trained SAE on feed-forward layer outputs (layers 40–70% of depth), compute per-token SAE activation entropy $H = -\sum_i p_i \log p_i$ (where $p_i$ are normalized feature activations) on TruthfulQA and FactScore data.
2. Align each token-level entropy estimate with ground-truth hallucination labels (e.g., use FactScore's sentence-level hallucination annotations projected onto token positions).
3. Report AUC for entropy as a hallucination detector and compare to a baseline probe on raw activations.
4. As a causal probe: implement a simple "top-$k$ feature concentration" intervention (zero out all but the top-$k$ features by magnitude) at high-entropy tokens and measure change in FactScore precision on a held-out set.
*Estimated timeline: 8–10 weeks; requires FactScore-annotated outputs or a re-run of the model on FactScore prompts.*

---

# Convergence and Divergence

**Convergence.** Both hypotheses agree on three foundational points:
1. **Layer locality:** Hallucination-relevant SAE signals are concentrated in mid-to-late transformer layers, not uniformly distributed—consistent with the established view that factual recall is mediated by mid-network feed-forward modules.
2. **Sparsity is load-bearing:** Both rely on the sparse decomposition SAEs provide; dense probing on raw activations is explicitly not the mechanism of interest.
3. **Causal intervention is the goal:** Both frame hallucination detection as a precursor to inference-time correction, not merely offline labeling.

**Divergence.** The hypotheses diverge on *what feature state is pathological:*
- `[H-fb0242]` posits a specific positive feature state—a "confident-without-grounding" cluster—that should be *suppressed*.
- `[H-77724]` posits an *entropic* or *conflicted* state—multiple competing features co-activating—that should be *resolved* toward a single dominant feature.

These are not mutually exclusive: it is possible that "confident-without-grounding" states arise precisely when epistemic superposition is not resolved and the model defaults to a high-confidence wrong feature. However, the intervention logics are different and could produce opposing predictions. For instance, if the correct account is epistemic superposition, suppressing the "confident" feature cluster (Direction 1) might paradoxically reduce confidence without improving accuracy. Conversely, if the correct account is an identifiable positive hallucination feature cluster, entropy-based detection (Direction 2) may miss cases where the model is confidently wrong from the start (no conflict, just wrong). **Running both experiments in parallel on the same model and benchmark set would directly adjudicate between them.**

---

# Caveats and Limitations

**What the tournament did not explore:**
- **SAE quality and coverage:** Both hypotheses implicitly assume well-trained SAEs with high reconstruction fidelity and meaningful feature interpretability. In practice, SAE training on frontier-scale models (70B+) is expensive and the resulting features may not cleanly decompose into monosemantic units at every layer. The tournament did not evaluate this prerequisite.
- **Benchmark-specific confounds:** TruthfulQA is known to be gameable by refusal; FactScore and HaluEval test different hallucination types (open-domain factual generation vs. NLI-style factual consistency). Neither hypothesis carefully distinguishes these regimes, and the intervention logic may differ across them.
- **Multi-hop and compositional hallucinations:** Both directions focus implicitly on single-fact retrieval failures. Hallucinations that arise from multi-hop reasoning errors or compositional failures may not have a simple SAE-feature-level signature.
- **Cross-model generalizability:** SAE features are model-specific. The tournament did not address whether identified hallucination features transfer across model families or even across fine-tuning variants of the same base model.
- **Interaction with RLHF/instruction tuning:** Both hypotheses were framed in terms of base model activations. Instruction-tuned and RLHF-trained models may have substantially different internal representations of "confidence" and factual uncertainty, which could invalidate the feature clusters identified on base models.

**Where the literature was thin:**
The tournament relied on literature-strategy hypotheses but no reviews with explicit citations were returned, suggesting the search may not have surfaced the most directly relevant recent work on SAE-based mechanistic analysis of factual recall (e.g., work on "knowledge neurons" in feed-forward layers, or recent Anthropic/EleutherAI SAE scaling analyses). A domain expert would likely ask for grounding in those specific results before accepting the layer-locality claims.

**Where a domain expert might most disagree:**
- The assumption that SAE features are sufficiently monosemantic at frontier scale to identify a "hallucination feature cluster" is contested—recent work on superposition suggests that even SAE-trained features may remain entangled at high parameter counts.
- The "epistemic superposition" framing in `[H-77724]` is evocative but may anthropomorphize an information-processing phenomenon that has a more mundane explanation (e.g., the model simply has low-rank uncertainty in its weight matrices, not a rich "belief conflict" structure).
- Inference-time intervention via activation steering has shown mixed results in prior work; gains on one benchmark often come with regressions elsewhere, and this tradeoff was not addressed in either hypothesis.