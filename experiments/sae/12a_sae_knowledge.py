#!/usr/bin/env python3
"""
Example 12: SAE Knowledge Assessment on GPT-2

Demonstrates the RO Framework assessing graded knowledge K(d_ext) = (ρ, ε, σ, C)
on real learned features from GPT-2, using pre-trained Sparse Autoencoders (SAEs).

External DoFs are abstract labels (sentiment, is_code).
Internal DoFs are SAE feature activations at specific layers.
Knowledge assessment tells us which SAE features track each label,
and with what correlation, bias, noise, and calibration.

This is richer than a linear probe: instead of "85% accuracy on sentiment",
you get a four-dimensional profile per feature.

Requires: pip install ro-framework[sae]
First run downloads GPT-2 small (~500MB) and SAE weights (~75MB per layer).
"""

import time

import numpy as np
import torch

from transformer_lens import HookedTransformer
from sae_lens import SAE

from ro_framework.core.dof import PolarDoF, ScalarDoF
from ro_framework.integration.sae import SAEObserver, create_feature_dofs

# ── Configuration ──────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
LAYERS = [0, 4, 8, 11]  # layers to compare
PRIMARY_LAYER = 8  # layer for detailed analysis

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print()

# ── Define external DoFs (labels) ─────────────────────────────────────

sentiment_dof = PolarDoF(
    name="sentiment",
    description="Positive/negative sentiment",
    pole_negative=-1.0,
    pole_positive=1.0,
)
is_code_dof = ScalarDoF(
    name="is_code",
    description="Whether text is code (0) or prose (1)",
    min_value=0.0,
    max_value=1.0,
)

label_dofs = [sentiment_dof, is_code_dof]

# ── Prepare labeled data ──────────────────────────────────────────────

# Sentiment-labeled texts (positive sentiment → +1, negative → -1)
positive_texts = [
    "I absolutely loved this movie, it was fantastic and beautiful",
    "This restaurant serves the most delicious food I've ever tasted",
    "What a wonderful day, everything went perfectly",
    "The performance was outstanding and truly inspiring",
    "I'm so happy with this purchase, highly recommended",
    "Beautiful sunset over the ocean, breathtaking scenery",
    "The team did an amazing job on this project",
    "Such a heartwarming story, it made me smile all day",
    "Excellent service, friendly staff, great experience overall",
    "This book changed my life, profound and moving",
    "The concert was incredible, best live music I've heard",
    "I feel grateful and blessed to have such wonderful friends",
    "The garden looks absolutely gorgeous this spring",
    "Brilliant solution to a difficult problem, very clever",
    "The children were laughing and playing, pure joy",
    "A masterpiece of cinema, every scene was perfect",
    "The cake was divine, moist and perfectly balanced flavors",
    "What an achievement, truly remarkable dedication and skill",
    "The sunrise was spectacular, golden light everywhere",
    "I'm thrilled with the results, exceeded all expectations",
]

negative_texts = [
    "This movie was terrible, worst I've seen in years",
    "The food was disgusting and the service was awful",
    "What a horrible day, everything went wrong",
    "The performance was disappointing and boring",
    "Waste of money, completely useless product",
    "Ugly weather, gray skies and freezing cold rain",
    "The team failed miserably on this project",
    "Such a depressing story, it ruined my whole evening",
    "Terrible service, rude staff, never going back",
    "This book was a waste of time, poorly written garbage",
    "The concert was dreadful, couldn't hear anything properly",
    "I feel lonely and isolated from everyone around me",
    "The garden is dead and overgrown with weeds",
    "Stupid mistake that caused enormous problems for everyone",
    "The children were crying and fighting, total chaos",
    "A disaster of a film, incoherent plot and bad acting",
    "The cake was stale and tasteless, very disappointing",
    "What a failure, wasted years of effort for nothing",
    "The storm was devastating, widespread damage everywhere",
    "I'm frustrated with the results, complete disappointment",
]

# Code samples
code_texts = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "import numpy as np; x = np.array([1, 2, 3]); print(x.mean())",
    "class MyModel(nn.Module): def forward(self, x): return self.linear(x)",
    "for i in range(10): print(f'iteration {i}')",
    "with open('data.json', 'r') as f: data = json.load(f)",
    "result = [x**2 for x in range(100) if x % 3 == 0]",
    "async def fetch_data(url): async with aiohttp.get(url) as resp: return await resp.json()",
    "df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}); df.groupby('a').sum()",
    "SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 5",
    "git commit -m 'fix: resolve null pointer in auth module'",
    "docker run -d --name myapp -p 8080:80 nginx:latest",
    "const express = require('express'); const app = express(); app.listen(3000)",
    "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')",
    "fn main() { let mut v = vec![1, 2, 3]; v.push(4); println!(\"{:?}\", v); }",
    "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100) NOT NULL)",
    "kubectl apply -f deployment.yaml && kubectl get pods",
    "from flask import Flask; app = Flask(__name__); @app.route('/') def hello(): return 'Hi'",
    "std::vector<int> v = {1, 2, 3}; std::sort(v.begin(), v.end());",
    "pip install transformers torch numpy scipy scikit-learn",
    "export CUDA_VISIBLE_DEVICES=0; python train.py --epochs 100 --lr 0.001",
]

# Build labeled dataset
dataset = []
for text in positive_texts:
    dataset.append((text, {sentiment_dof: 0.9, is_code_dof: 0.0}))
for text in negative_texts:
    dataset.append((text, {sentiment_dof: -0.9, is_code_dof: 0.0}))
for text in code_texts:
    dataset.append((text, {sentiment_dof: 0.0, is_code_dof: 1.0}))

np.random.seed(42)
indices = np.random.permutation(len(dataset))
dataset = [dataset[i] for i in indices]

print(f"Dataset: {len(dataset)} texts ({len(positive_texts)} positive, "
      f"{len(negative_texts)} negative, {len(code_texts)} code)")
print()

# ── Load model ─────────────────────────────────────────────────────────

print("Loading GPT-2 small...")
t0 = time.time()
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
print(f"  Model loaded in {time.time() - t0:.1f}s")

# ── Part 1: Detailed analysis at primary layer ────────────────────────

print(f"\n{'='*60}")
print(f"Part 1: SAE Knowledge Assessment at Layer {PRIMARY_LAYER}")
print(f"{'='*60}\n")

print(f"Loading SAE for layer {PRIMARY_LAYER}...")
t0 = time.time()
hook_point = f"blocks.{PRIMARY_LAYER}.hook_resid_pre"
sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=hook_point,
    device=DEVICE,
)
if isinstance(sae, tuple):
    sae = sae[0]
print(f"  SAE loaded in {time.time() - t0:.1f}s (d_sae={sae.cfg.d_sae})")

# Create SAEObserver
observer = SAEObserver(
    model=model,
    sae=sae,
    hook_point=hook_point,
    label_dofs=label_dofs,
    name=f"gpt2_L{PRIMARY_LAYER}",
    device=DEVICE,
)

# Feed all texts
print(f"\nProcessing {len(dataset)} texts...")
t0 = time.time()
for text, labels in dataset:
    observer.observe_text(text, labels)
print(f"  Done in {time.time() - t0:.1f}s ({observer.n_observations} observations)")

# Assess knowledge of sentiment
print(f"\n--- Sentiment Knowledge (Layer {PRIMARY_LAYER}) ---\n")
k_sentiment = observer.assess_knowledge(sentiment_dof)
if k_sentiment:
    print(f"  Best feature: {k_sentiment.best_internal_dof.name}")
    print(f"  Correlation (ρ): {k_sentiment.correlation:.3f}")
    print(f"  Systematic error (ε): {k_sentiment.systematic_error:+.3f}")
    print(f"  Random error (σ): {k_sentiment.random_error:.3f}")
    print(f"  Calibration (C): {k_sentiment.calibration:.3f}")
    print(f"  Knowledge type: {k_sentiment.knowledge_type}")

print(f"\n  Top 10 features correlated with sentiment:")
print(f"  {'Feature':<20} {'ρ':>6} {'ε':>8} {'σ':>6} {'C':>6}  {'Type':<10}")
print(f"  {'-'*18}  {'-'*5} {'-'*7} {'-'*5} {'-'*5}  {'-'*9}")
top_sentiment = observer.top_features_for(sentiment_dof, n=10)
for fk in top_sentiment:
    k = fk.assessment
    print(f"  {fk.feature_dof.name:<20} {k.correlation:>6.3f} {k.systematic_error:>+8.3f} "
          f"{k.random_error:>6.3f} {k.calibration:>6.3f}  {k.knowledge_type:<10}")

# Assess knowledge of is_code
print(f"\n--- Code Detection Knowledge (Layer {PRIMARY_LAYER}) ---\n")
k_code = observer.assess_knowledge(is_code_dof)
if k_code:
    print(f"  Best feature: {k_code.best_internal_dof.name}")
    print(f"  Correlation (ρ): {k_code.correlation:.3f}")
    print(f"  Systematic error (ε): {k_code.systematic_error:+.3f}")
    print(f"  Random error (σ): {k_code.random_error:.3f}")
    print(f"  Calibration (C): {k_code.calibration:.3f}")
    print(f"  Knowledge type: {k_code.knowledge_type}")

print(f"\n  Top 10 features correlated with is_code:")
print(f"  {'Feature':<20} {'ρ':>6} {'ε':>8} {'σ':>6} {'C':>6}  {'Type':<10}")
print(f"  {'-'*18}  {'-'*5} {'-'*7} {'-'*5} {'-'*5}  {'-'*9}")
top_code = observer.top_features_for(is_code_dof, n=10)
for fk in top_code:
    k = fk.assessment
    print(f"  {fk.feature_dof.name:<20} {k.correlation:>6.3f} {k.systematic_error:>+8.3f} "
          f"{k.random_error:>6.3f} {k.calibration:>6.3f}  {k.knowledge_type:<10}")

# ── Part 2: Multi-layer comparison ────────────────────────────────────

print(f"\n{'='*60}")
print(f"Part 2: Knowledge Across Layers (hierarchical decomposition)")
print(f"{'='*60}\n")

print(f"Loading SAEs for layers {LAYERS}...")
layer_observers = {}
for layer in LAYERS:
    hook = f"blocks.{layer}.hook_resid_pre"
    t0 = time.time()
    layer_sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=hook,
        device=DEVICE,
    )
    if isinstance(layer_sae, tuple):
        layer_sae = layer_sae[0]

    layer_obs = SAEObserver(
        model=model,
        sae=layer_sae,
        hook_point=hook,
        label_dofs=label_dofs,
        name=f"gpt2_L{layer}",
        device=DEVICE,
    )
    layer_observers[layer] = layer_obs
    print(f"  Layer {layer:>2}: loaded in {time.time() - t0:.1f}s (d_sae={layer_sae.cfg.d_sae})")

# Feed same data to all layers
print(f"\nProcessing {len(dataset)} texts across {len(LAYERS)} layers...")
t0 = time.time()
for text, labels in dataset:
    for layer_obs in layer_observers.values():
        layer_obs.observe_text(text, labels)
print(f"  Done in {time.time() - t0:.1f}s")

# Compare sentiment knowledge across depth
print(f"\n--- Sentiment Knowledge Across Layers ---\n")
print(f"  {'Layer':>5}  {'ρ':>6}  {'ε':>8}  {'σ':>6}  {'C':>6}  {'Type':<10}  Best Feature")
print(f"  {'-'*4}   {'-'*5}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*20}")
for layer in LAYERS:
    k = layer_observers[layer].assess_knowledge(sentiment_dof)
    if k:
        print(f"  {layer:>5}  {k.correlation:>6.3f}  {k.systematic_error:>+8.3f}  "
              f"{k.random_error:>6.3f}  {k.calibration:>6.3f}  {k.knowledge_type:<10}  "
              f"{k.best_internal_dof.name}")
    else:
        print(f"  {layer:>5}  insufficient data")

# Compare code detection across depth
print(f"\n--- Code Detection Knowledge Across Layers ---\n")
print(f"  {'Layer':>5}  {'ρ':>6}  {'ε':>8}  {'σ':>6}  {'C':>6}  {'Type':<10}  Best Feature")
print(f"  {'-'*4}   {'-'*5}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*20}")
for layer in LAYERS:
    k = layer_observers[layer].assess_knowledge(is_code_dof)
    if k:
        print(f"  {layer:>5}  {k.correlation:>6.3f}  {k.systematic_error:>+8.3f}  "
              f"{k.random_error:>6.3f}  {k.calibration:>6.3f}  {k.knowledge_type:<10}  "
              f"{k.best_internal_dof.name}")
    else:
        print(f"  {layer:>5}  insufficient data")

print(f"\n{'='*60}")
print("Done. Each K tuple tells you not just IF the model knows,")
print("but HOW it knows: with what bias, noise, and calibration.")
print(f"{'='*60}")
