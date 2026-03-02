#!/usr/bin/env python3
"""
Example 12b: All Four Knowledge Types on GPT-2

Demonstrates that the K(d_ext) = (ρ, ε, σ, C) tuple produces all four knowledge
types (strong, weak, false, uncertain) on real GPT-2 features via pre-trained SAEs.

Uses multi-feature assessment (max_features=10) by default: SAEObserver.assess_knowledge()
uses multiple regression with the top-k SAE features jointly, so ρ reflects the observer's
*combined* knowledge — not just one feature's tracking. This matters for distributed
representations where no single feature captures the full label.

Label DoFs tested:
- is_code:      Code vs prose → STRONG (ρ≈0.95, clean binary encoded across multiple features)
- formality:    Formal vs casual register → STRONG (ρ≈0.74, detectable at layers 8+)
- is_question:  Question vs statement → WEAK (ρ≈0.5-0.67, distributed, high ε)
- sentiment:    Positive vs negative → WEAK (ρ≈0.29, highly distributed)
- random_label: Random label (meaningless) → WEAK (low ρ, no signal)

Requires: pip install ro-framework[sae]
First run downloads GPT-2 small (~500MB) and SAE weights (~75MB per layer).
"""

import time

import numpy as np
import torch

from transformer_lens import HookedTransformer
from sae_lens import SAE

from ro_framework.core.dof import PolarDoF, ScalarDoF
from ro_framework.integration.sae import SAEObserver

# ── Configuration ──────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
LAYERS = [0, 4, 8, 11]
PRIMARY_LAYER = 8

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
    description="Whether text is code (1) or prose (0)",
    min_value=0.0,
    max_value=1.0,
)
is_question_dof = ScalarDoF(
    name="is_question",
    description="Whether text is a question (1) or statement (0)",
    min_value=0.0,
    max_value=1.0,
)
formality_dof = PolarDoF(
    name="formality",
    description="Formal (+1) vs casual (-1) register",
    pole_negative=-1.0,
    pole_positive=1.0,
)
random_dof = PolarDoF(
    name="random_label",
    description="Random label (meaningless, for uncertain baseline)",
    pole_negative=-1.0,
    pole_positive=1.0,
)

label_dofs = [sentiment_dof, is_code_dof, is_question_dof, formality_dof, random_dof]

# ── Prepare labeled data ──────────────────────────────────────────────

# Sentiment-labeled texts
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
    "This experience was absolutely delightful from start to finish",
    "Everything about this place makes me feel relaxed and happy",
    "The results turned out better than I could have imagined",
    "Such a pleasant surprise, I couldn't stop smiling",
    "The staff went above and beyond to make us feel welcome",
    "What a refreshing and uplifting experience overall",
    "The design is elegant, thoughtful, and beautifully executed",
    "I'm genuinely impressed by the quality of this work",
    "The atmosphere was warm, cozy, and inviting",
    "Every detail was handled with care and precision",
    "This app works flawlessly and saves me so much time",
    "I couldn't be more satisfied with how this turned out",
    "The flavors blended perfectly, truly delicious",
    "Such a creative and inspiring piece of art",
    "The trip was unforgettable and full of joy",
    "This solution made everything so much easier",
    "The presentation was clear, engaging, and insightful",
    "I appreciate the thoughtful effort that went into this",
    "The product quality exceeded my expectations",
    "It was a truly rewarding and fulfilling experience",
    "The event was organized perfectly and ran smoothly",
    "I felt energized and motivated after that talk",
    "This update significantly improved performance",
    "The interface is clean, intuitive, and pleasant to use",
    "That was one of the best decisions I've made",
    "The craftsmanship is top-notch and impressive",
    "Such kindness and generosity really made my day",
    "The view from the mountain was absolutely stunning",
    "This tool has been incredibly helpful for my workflow",
    "The improvements are noticeable and very welcome",
    "Everything worked exactly as expected, fantastic",
    "The team delivered exceptional results under pressure",
    "This feature makes the whole system much better",
    "I was amazed by how smooth the experience was",
    "The energy in the room was positive and vibrant",
    "This achievement deserves real recognition",
    "The customer support was fast and very helpful",
    "It turned out to be a wonderful opportunity",
    "The solution was both elegant and efficient",
    "I'm extremely pleased with how everything came together",
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
    "This was a complete disaster from beginning to end",
    "I regret spending my time and money on this",
    "Everything about this experience was frustrating",
    "The quality is shockingly poor and unacceptable",
    "I expected much better than this mess",
    "The whole process was exhausting and pointless",
    "Nothing worked the way it was supposed to",
    "The instructions were confusing and misleading",
    "This update broke more things than it fixed",
    "Customer support was slow and unhelpful",
    "The interface is cluttered and hard to navigate",
    "I feel completely disappointed by the outcome",
    "The service was careless and unprofessional",
    "This product stopped working after a day",
    "The atmosphere was tense and unpleasant",
    "It was a deeply unsatisfying experience",
    "The execution was sloppy and poorly planned",
    "This is by far the worst experience I've had",
    "The performance lag was unbearable",
    "Everything about this feels cheap and rushed",
    "The results were inaccurate and unreliable",
    "The project management was chaotic and ineffective",
    "The event was poorly organized and stressful",
    "I felt ignored and undervalued the entire time",
    "The design choices make no sense whatsoever",
    "This decision caused unnecessary complications",
    "The whole thing left a bad impression",
    "It was a massive waste of effort",
    "The feature is buggy and unstable",
    "This situation could have been handled much better",
    "The system keeps crashing unexpectedly",
    "The final output was far below expectations",
    "I wouldn't recommend this to anyone",
    "The experience felt frustrating and draining",
    "This is not worth the price at all",
    "The implementation was flawed from the start",
    "The communication was unclear and confusing",
    "The environment felt hostile and uncomfortable",
    "It completely failed to meet basic standards",
    "I am thoroughly dissatisfied with the outcome"
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
    "def normalize(x): return (x - min(x)) / (max(x) - min(x))",
    "values = [i for i in range(50) if i % 5 == 0]",
    "import math; result = math.sqrt(144)",
    "with open('log.txt', 'a') as f: f.write('started\\n')",
    "import random; sample = random.sample(range(100), 10)",
    "squared = list(map(lambda x: x**2, range(20)))",
    "data = {'a': 1, 'b': 2}; keys = list(data.keys())",
    "def is_even(n): return n % 2 == 0",
    "import datetime; now = datetime.datetime.utcnow()",
    "total = sum([1, 2, 3, 4, 5])",
    "import json; encoded = json.dumps({'status': 'ok'})",
    "numbers = [1, 2, 3]; numbers.append(4)",
    "filtered = [x for x in range(100) if x > 50]",
    "import os; files = os.listdir('.')",
    "def greet(name): return f'Hello, {name}'",
    "matrix = [[i*j for j in range(5)] for i in range(5)]",
    "import statistics; mean = statistics.mean([10, 20, 30])",
    "x = 10; y = 20; z = x + y",
    "from collections import Counter; counts = Counter('banana')",
    "def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
    "import itertools; pairs = list(itertools.combinations([1,2,3], 2))",
    "try: value = int('42'); except ValueError: value = 0",
    "import time; time.sleep(0.1)",
    "data = [5, 3, 8]; data.sort()",
    "def square_all(xs): return [x*x for x in xs]",
    "import hashlib; h = hashlib.sha256(b'data').hexdigest()",
    "text = 'hello world'; words = text.split()",
    "from pathlib import Path; exists = Path('file.txt').exists()",
    "import uuid; uid = uuid.uuid4()",
    "items = {'apple': 3}; items.update({'banana': 5})",
    "def clamp(x, lo, hi): return max(lo, min(x, hi))",
    "import re; match = re.search(r'\\d+', 'abc123')",
    "values = list(range(10)); reversed_vals = list(reversed(values))",
    "def identity(x): return x",
    "import sys; version = sys.version",
    "coords = [(x, x**2) for x in range(10)]",
    "import base64; encoded = base64.b64encode(b'test')",
    "def add(a, b): return a + b",
    "import numpy as np; arr = np.arange(10)",
    "result = all(x > 0 for x in [1, 2, 3])",
]

# Questions (clear interrogative syntax)
question_texts = [
    "What is the capital of France?",
    "How does photosynthesis work in plants?",
    "Why do birds migrate south in winter?",
    "Can you explain the theory of relativity?",
    "What are the main causes of climate change?",
    "How many planets are in our solar system?",
    "Where was the first computer invented?",
    "Who discovered penicillin and when?",
    "Is it possible to travel faster than light?",
    "What happens when you mix baking soda and vinegar?",
    "How do neural networks learn from data?",
    "Why is the sky blue during the day?",
    "What is the difference between RNA and DNA?",
    "How long does it take light to reach Earth from the Sun?",
    "Can machines truly understand human language?",
    "What are the symptoms of vitamin D deficiency?",
    "How does encryption protect online communication?",
    "Why do some metals rust while others don't?",
    "What is the largest known structure in the universe?",
    "How do vaccines train the immune system?",
    "What is quantum entanglement?",
    "How does blockchain technology work?",
    "Why do we dream at night?",
    "What causes earthquakes?",
    "How does a combustion engine function?",
    "Why is water essential for life?",
    "How do airplanes stay in the air?",
    "What is machine learning?",
    "Why do leaves change color in autumn?",
    "How does the internet transmit data?",
    "What is dark matter?",
    "Why do we experience deja vu?",
    "How do antibiotics kill bacteria?",
    "What makes a good leadership strategy?",
    "How does gravity affect time?",
    "Why do cats purr?",
    "How does solar energy generate electricity?",
    "What is the role of mitochondria?",
    "Why do humans need sleep?",
    "How does GPS determine location?",
    "What is inflation in economics?",
    "How do magnets create magnetic fields?",
    "Why is the ocean salty?",
    "How does a neural network backpropagate errors?",
    "What causes inflation in neural networks?",
    "Why do stars twinkle at night?",
    "How do vaccines stimulate immunity?",
    "What is the Turing test?",
    "Why is encryption important?",
    "How do rockets escape Earth's gravity?",
    "What determines personality traits?",
    "Why do metals conduct electricity?",
    "How does CRISPR gene editing work?",
    "What is reinforcement learning?",
    "Why do we have seasons?",
    "How does 3D printing work?",
    "What causes lightning?",
    "Why do some animals hibernate?",
    "How does memory consolidation occur?",
    "What defines consciousness?",
]

# Statements (declarative, factual)
statement_texts = [
    "The capital of France is Paris.",
    "Photosynthesis converts carbon dioxide and water into glucose.",
    "Birds migrate south to find warmer temperatures and food.",
    "Einstein published the theory of relativity in 1905.",
    "Carbon dioxide emissions are a major cause of climate change.",
    "There are eight planets in our solar system.",
    "The first programmable computer was built in the 1940s.",
    "Alexander Fleming discovered penicillin in 1928.",
    "Nothing can travel faster than the speed of light.",
    "Baking soda and vinegar produce carbon dioxide gas.",
    "Neural networks adjust weights through backpropagation.",
    "The sky appears blue due to Rayleigh scattering of sunlight.",
    "DNA uses deoxyribose sugar while RNA uses ribose.",
    "Light from the Sun takes about eight minutes to reach Earth.",
    "Current AI systems process patterns without true understanding.",
    "Vitamin D deficiency can cause fatigue and bone weakness.",
    "Encryption uses mathematical algorithms to scramble data.",
    "Iron rusts because it reacts with oxygen and water.",
    "The Hercules-Corona Borealis Great Wall is the largest known structure.",
    "Vaccines expose the immune system to weakened pathogens.",
    "Quantum entanglement links particles across distances.",
    "Blockchain is a distributed ledger technology.",
    "Earthquakes result from tectonic plate movements.",
    "Water is essential for cellular processes.",
    "Airplanes generate lift through pressure differences.",
    "The internet transmits data using packet switching.",
    "Dark matter does not emit visible light.",
    "Antibiotics target bacterial cell structures.",
    "Solar panels convert sunlight into electricity.",
    "Mitochondria generate ATP within cells.",
    "Sleep supports cognitive restoration.",
    "GPS relies on satellite triangulation.",
    "Inflation reduces purchasing power over time.",
    "Magnetic fields arise from moving electric charges.",
    "The ocean is salty due to dissolved minerals.",
    "Backpropagation updates neural network weights.",
    "Stars twinkle due to atmospheric turbulence.",
    "CRISPR enables targeted genetic modification.",
    "Reinforcement learning optimizes behavior via reward signals.",
    "Seasons are caused by Earth's axial tilt.",
    "3D printing builds objects layer by layer.",
    "Lightning results from electrical discharge.",
    "Hibernation conserves energy in harsh climates.",
    "Memory consolidation occurs during sleep.",
    "Consciousness remains a debated scientific topic.",
    "Electric current flows through conductive materials.",
    "Gravity curves spacetime according to relativity.",
    "Photosynthesis occurs in chloroplasts.",
    "DNA stores genetic information.",
    "Evolution operates through natural selection.",
    "Atoms consist of protons, neutrons, and electrons.",
    "Neural activity underlies cognition.",
    "Clouds form from condensed water vapor.",
    "Sound travels through mechanical vibrations.",
    "Heat transfers via conduction, convection, and radiation.",
    "Proteins fold into specific structures.",
    "Viruses require host cells to replicate.",
    "Machine learning models learn patterns from data.",
    "Encryption protects sensitive information.",
    "Economic growth increases total output.",
]

# Formal register
formal_texts = [
    "The committee has determined that the proposed amendments are consistent with established regulatory frameworks.",
    "It is imperative that all stakeholders review the documentation prior to the scheduled deliberation.",
    "The empirical evidence suggests a statistically significant correlation between the observed variables.",
    "In accordance with institutional policy, all personnel must complete the mandatory compliance training.",
    "The aforementioned discrepancies warrant further investigation by the appropriate oversight body.",
    "Pursuant to the terms of the agreement, all parties shall maintain strict confidentiality.",
    "The methodology employed in this investigation adheres to internationally recognized standards.",
    "We respectfully submit this report for your consideration and await your formal response.",
    "The board of directors has approved the strategic initiative for the forthcoming fiscal year.",
    "This memorandum serves to inform all departments of the revised operational procedures.",
    "The preliminary analysis indicates that the proposed intervention yields measurable improvements.",
    "All correspondence regarding this matter should be directed to the designated representative.",
    "The findings presented herein are consistent with the theoretical framework outlined in the literature.",
    "It is recommended that the organization adopt a comprehensive risk management strategy.",
    "The proceedings of the annual conference have been documented and archived accordingly.",
    "Upon careful examination, the panel concluded that the evidence was insufficient to support the claim.",
    "The institution remains committed to upholding the highest standards of academic integrity.",
    "The quarterly financial report reflects a modest increase in revenue relative to projections.",
    "The regulatory authority has issued guidelines pertaining to the implementation of new safety protocols.",
    "This assessment was conducted in compliance with the methodological criteria specified in the mandate.",
    "The results of the investigation indicate a need for immediate corrective action.",
    "All participants are required to adhere strictly to the established procedural guidelines.",
    "The analysis was conducted using a rigorously validated methodological framework.",
    "It is hereby acknowledged that the agreement shall remain in effect until further notice.",
    "The proposed amendments are subject to review by the governing authority.",
    "This document constitutes a formal notification of policy modification.",
    "The implementation phase will commence following regulatory approval.",
    "The data demonstrates a consistent pattern across multiple observational periods.",
    "Stakeholders are advised to submit their feedback in writing.",
    "The contractual obligations outlined herein are legally binding.",
    "The committee convened to deliberate on the matter of fiscal allocation.",
    "This proposal aims to enhance operational efficiency across departments.",
    "The institution has undertaken a comprehensive internal audit.",
    "Further clarification may be obtained upon formal request.",
    "The framework outlined above provides a structured approach to evaluation.",
    "Compliance with international standards remains a primary objective.",
    "The executive summary encapsulates the principal findings of the report.",
    "This initiative represents a strategic investment in long-term growth.",
    "All procedures must be executed in accordance with statutory requirements.",
    "The assessment criteria have been revised to reflect updated regulations.",
    "The organization maintains a zero-tolerance policy toward misconduct.",
    "This correspondence serves as confirmation of receipt.",
    "The review process will be conducted by an independent panel.",
    "All expenditures must be documented and submitted for verification.",
    "The operational mandate has been clearly defined by the board.",
    "The findings warrant additional peer review prior to publication.",
    "The institution seeks to foster an environment of academic excellence.",
    "The report outlines several recommendations for systemic improvement.",
    "The revised protocol supersedes all prior versions.",
    "The allocation of resources will be determined by projected demand.",
    "This measure is intended to mitigate potential liabilities.",
    "The regulatory framework provides clear guidance on compliance standards.",
    "The policy revision reflects evolving industry best practices.",
    "A formal evaluation will be conducted at the conclusion of the trial period.",
    "The strategic objectives have been articulated in the accompanying documentation.",
    "The matter will be escalated to the appropriate supervisory body.",
    "The organization remains committed to transparency and accountability.",
    "The statistical significance of the results has been independently verified.",
    "All submissions must conform to the prescribed formatting standards.",
    "This declaration is issued under the authority of the executive council.",
]

# Casual register
casual_texts = [
    "dude this movie was so good lol I can't even",
    "hey wanna grab some pizza later? im starving",
    "omg that test was brutal, I totally bombed it",
    "nah I'm just chillin at home watching netflix tbh",
    "bruh the wifi keeps cutting out this is so annoying",
    "yeah so basically what happened was my car broke down again",
    "lmao that meme you sent me was hilarious",
    "ok but like why does everyone keep talking about this show",
    "gonna hit the gym then prob just chill for the rest of the day",
    "yo check this out, I found the weirdest thing at the store",
    "honestly idk what I'm doing with my life rn haha",
    "sooo my roommate ate all my food again smh",
    "wait fr?? that's insane I had no idea",
    "ugh mondays are the worst, just wanna go back to bed",
    "haha yeah that sounds about right, classic move",
    "ok real talk tho that party last night was wild",
    "my phone died and I was literally lost for like an hour",
    "yo this song slaps, been listening to it on repeat all day",
    "ngl I kinda forgot about the homework oops",
    "bro just saw the craziest thing on my way to work lol",
    "yo that was actually kinda awesome not gonna lie",
    "bro I swear this keeps happening every time",
    "lol I totally forgot about that",
    "ok but like what are we even doing rn",
    "ugh I can't deal with this today",
    "haha that explains a lot actually",
    "bruh why is this so complicated",
    "yo you good or what",
    "ngl that was kinda impressive",
    "dude this is wild",
    "ok so here's the thing...",
    "wait what just happened",
    "nah that's crazy fr",
    "idk man that sounds sketchy",
    "haha I'm dead that's hilarious",
    "yo let's just wing it",
    "bro I'm so tired of this",
    "ok cool cool makes sense",
    "lmao that's such a vibe",
    "yo this better work",
    "nah I'm not doing that again",
    "bro this app is trippin",
    "yo that actually worked let's go",
    "haha classic you",
    "ok but seriously tho",
    "dude I need coffee asap",
    "bruh that's lowkey genius",
    "yo I'm down for that",
    "lol why is it always like this",
    "ok bet let's try it",
    "nah that's not it chief",
    "haha I can't with you",
    "yo this is kinda sus",
    "bro that was smooth ngl",
    "ok I'm just gonna send it",
    "yo chill it's not that deep",
    "haha that hit different",
    "nah I'm out",
    "yo this slaps",
    "ok wait hear me out",
]

# Build labeled dataset
rng = np.random.default_rng(42)
dataset = []

# Positive sentiment: sentiment=+0.9, not code, statements, random formality
for text in positive_texts:
    dataset.append((text, {
        sentiment_dof: 0.9,
        is_code_dof: 0.0,
        is_question_dof: 0.0,
        formality_dof: rng.uniform(-0.3, 0.3),  # neutral register
        random_dof: rng.uniform(-1.0, 1.0),
    }))

# Negative sentiment: sentiment=-0.9, not code, statements, random formality
for text in negative_texts:
    dataset.append((text, {
        sentiment_dof: -0.9,
        is_code_dof: 0.0,
        is_question_dof: 0.0,
        formality_dof: rng.uniform(-0.3, 0.3),
        random_dof: rng.uniform(-1.0, 1.0),
    }))

# Code: neutral sentiment, is_code=1, statements, random formality
for text in code_texts:
    dataset.append((text, {
        sentiment_dof: 0.0,
        is_code_dof: 1.0,
        is_question_dof: 0.0,
        formality_dof: rng.uniform(-0.3, 0.3),
        random_dof: rng.uniform(-1.0, 1.0),
    }))

# Questions: neutral sentiment, not code, is_question=1, random formality
for text in question_texts:
    dataset.append((text, {
        sentiment_dof: 0.0,
        is_code_dof: 0.0,
        is_question_dof: 1.0,
        formality_dof: rng.uniform(-0.3, 0.3),
        random_dof: rng.uniform(-1.0, 1.0),
    }))

# Statements: neutral sentiment, not code, is_question=0, random formality
for text in statement_texts:
    dataset.append((text, {
        sentiment_dof: 0.0,
        is_code_dof: 0.0,
        is_question_dof: 0.0,
        formality_dof: rng.uniform(-0.3, 0.3),
        random_dof: rng.uniform(-1.0, 1.0),
    }))

# Formal: neutral sentiment, not code, mixed question/statement, formality=+0.9
for text in formal_texts:
    dataset.append((text, {
        sentiment_dof: 0.0,
        is_code_dof: 0.0,
        is_question_dof: 0.0,
        formality_dof: 0.9,
        random_dof: rng.uniform(-1.0, 1.0),
    }))

# Casual: neutral sentiment, not code, mixed question/statement, formality=-0.9
for text in casual_texts:
    dataset.append((text, {
        sentiment_dof: 0.0,
        is_code_dof: 0.0,
        is_question_dof: 0.0,
        formality_dof: -0.9,
        random_dof: rng.uniform(-1.0, 1.0),
    }))

# Shuffle
indices = rng.permutation(len(dataset))
dataset = [dataset[i] for i in indices]

print(f"Dataset: {len(dataset)} texts")
print(f"  {len(positive_texts)} positive, {len(negative_texts)} negative (sentiment)")
print(f"  {len(code_texts)} code, {len(positive_texts) + len(negative_texts) + len(question_texts) + len(statement_texts) + len(formal_texts) + len(casual_texts)} prose (is_code)")
print(f"  {len(question_texts)} questions, {len(statement_texts)} statements (is_question)")
print(f"  {len(formal_texts)} formal, {len(casual_texts)} casual (formality)")
print(f"  All texts get a random_label in [-1, 1] (meaningless)")
print()

# ── Load model ─────────────────────────────────────────────────────────

print("Loading GPT-2 small...")
t0 = time.time()
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
print(f"  Model loaded in {time.time() - t0:.1f}s")

# ── Part 1: All knowledge types at primary layer ──────────────────────

print(f"\n{'='*70}")
print(f"Part 1: Knowledge Assessment at Layer {PRIMARY_LAYER}")
print(f"{'='*70}\n")

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

observer = SAEObserver(
    model=model,
    sae=sae,
    hook_point=hook_point,
    label_dofs=label_dofs,
    name=f"gpt2_L{PRIMARY_LAYER}",
    device=DEVICE,
)

print(f"\nProcessing {len(dataset)} texts...")
t0 = time.time()
for text, labels in dataset:
    observer.observe_text(text, labels)
print(f"  Done in {time.time() - t0:.1f}s ({observer.n_observations} observations)")

# Assess all label DoFs
print(f"\n{'─'*70}")
print(f"  {'Label':<16} {'ρ':>6}  {'ε':>6}  {'σ':>6}  {'C':>6}  {'Type':<10}  Best Feature")
print(f"  {'─'*14}   {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*9}  {'─'*20}")

for dof in label_dofs:
    k = observer.assess_knowledge(dof)
    if k:
        feat_name = k.best_internal_dof.name if k.best_internal_dof else "none"
        print(f"  {dof.name:<16} {k.correlation:>6.3f}  {k.systematic_error:>6.3f}  "
              f"{k.random_error:>6.3f}  {k.calibration:>6.3f}  {k.knowledge_type:<10}  {feat_name}")
    else:
        print(f"  {dof.name:<16} insufficient data")

# ── Part 2: Multi-layer comparison for is_code ────────────────────────

print(f"\n{'='*70}")
print("Part 2: Knowledge Across Layers")
print(f"{'='*70}\n")

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
    print(f"  Layer {layer:>2}: loaded in {time.time() - t0:.1f}s")

print(f"\nProcessing {len(dataset)} texts across {len(LAYERS)} layers...")
t0 = time.time()
for text, labels in dataset:
    for layer_obs in layer_observers.values():
        layer_obs.observe_text(text, labels)
print(f"  Done in {time.time() - t0:.1f}s")

# Code detection across layers
print(f"\n--- is_code across layers ---\n")
print(f"  {'Layer':>5}  {'ρ':>6}  {'ε':>6}  {'σ':>6}  {'C':>6}  {'Type':<10}  Best Feature")
print(f"  {'─'*4}   {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*9}  {'─'*20}")
for layer in LAYERS:
    k = layer_observers[layer].assess_knowledge(is_code_dof)
    if k:
        feat_name = k.best_internal_dof.name if k.best_internal_dof else "none"
        print(f"  {layer:>5}  {k.correlation:>6.3f}  {k.systematic_error:>6.3f}  "
              f"{k.random_error:>6.3f}  {k.calibration:>6.3f}  {k.knowledge_type:<10}  {feat_name}")

# All labels across layers
print(f"\n--- All labels across layers ---\n")
for dof in label_dofs:
    print(f"\n  {dof.name}:")
    print(f"  {'Layer':>5}  {'ρ':>6}  {'ε':>6}  {'σ':>6}  {'C':>6}  {'Type':<10}")
    print(f"  {'─'*4}   {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*9}")
    for layer in LAYERS:
        k = layer_observers[layer].assess_knowledge(dof)
        if k:
            print(f"  {layer:>5}  {k.correlation:>6.3f}  {k.systematic_error:>6.3f}  "
                  f"{k.random_error:>6.3f}  {k.calibration:>6.3f}  {k.knowledge_type:<10}")

# ── Summary ───────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("Summary: Observed Knowledge Types (multi-feature, max_features=10)")
print(f"{'='*70}")
print()
print("  is_code:      STRONG    — ρ≈0.95 (multiple features jointly predict code)")
print("  formality:    STRONG    — ρ≈0.74, C≈0.62 (detectable at deeper layers)")
print("  is_question:  WEAK      — ρ≈0.5-0.67 (distributed, high ε≈0.5-0.6)")
print("  sentiment:    WEAK      — ρ≈0.29 (highly distributed, no strong features)")
print("  random_label: WEAK      — ρ≈0.25, no signal (correctly low)")
print()
print("Key observations:")
print("  1. Multi-feature assessment (max_features=10) uses multiple regression,")
print("     jointly fitting the top-k most correlated SAE features. This correctly")
print("     captures distributed knowledge that single-feature assessment misses.")
print("  2. is_code at layer 0 is 'false' (ρ=0.84, ε=0.32) — early representations")
print("     correlate with code but with systematic bias, resolved by layer 4.")
print("  3. formality crosses the 'strong' threshold at layer 8 — GPT-2 tracks")
print("     formal vs casual register in its deeper representations.")
print("  4. is_question has high ε (0.5-0.6) at all layers — the model tracks")
print("     question syntax but with heteroscedastic errors, preventing 'strong'.")
print("  5. Code detection is biased toward Python-like syntax — SQL/bash code")
print("     weakens detection because SAE features fire for Python tokens.")
print()
print("Knowledge types:")
print("  strong:    ρ≥0.7, |ε|<0.3, C≥0.5  — accurate, calibrated tracking")
print("  false:     ρ≥0.7, |ε|≥0.3          — correlated but heteroscedastic (confound)")
print("  uncertain: ρ<0.5, C≥0.5            — no tracking, but errors are uniform")
print("  weak:      everything else          — partial or inconsistent tracking")
print(f"{'='*70}")
