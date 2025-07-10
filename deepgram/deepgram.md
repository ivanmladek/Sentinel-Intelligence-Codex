Company Overview
Deepgram is the leading voice AI platform for developers building speech-to-text (STT), text-to-speech (TTS) and full speech-to-speech (STS) offerings. 200,000+ developers build with Deepgram’s voice-native foundational models – accessed through APIs or as self-managed software – due to our unmatched accuracy, latency and pricing. Customers include software companies building voice products, co-sell partners working with large enterprises, and enterprises solving internal voice AI use cases. The company ended 2024 cash-flow positive with 400+ enterprise customers, 3.3x annual usage growth across the past 4 years, over 50,000 years of audio processed and over 1 trillion words transcribed. There is no organization in the world that understands voice better than Deepgram
The Opportunity
Voice is the most natural modality for human interaction with machines. However, current sequence modeling paradigms based on jointly scaling model and data cannot deliver voice AI capable of universal human interaction. The challenges are rooted in fundamental data problems posed by audio: real-world audio data is scarce and enormously diverse, spanning a vast space of voices, speaking styles, and acoustic conditions. Even if billions of hours of audio were accessible, its inherent high dimensionality creates computational and storage costs that make training and deployment prohibitively expensive at world scale. We believe that entirely new paradigms for audio AI are needed to overcome these challenges and make voice interaction accessible to everyone.

The Role
You will pioneer the development of Latent Space Models (LSMs), a new approach that aims to solve the fundamental data, scale, and cost challenges associated with building robust, contextualized voice AI. Your research will focus on solving one or more of the following problems:
Build next-generation neural audio codecs that achieve extreme, low bit-rate compression and high fidelity reconstruction across a world-scale corpus of general audio.
Pioneer steerable generative models that can synthesize the full diversity of human speech from the codec latent representation, from casual conversation to highly emotional expression to complex multi-speaker scenarios with environmental noise and overlapping speech.
Develop embedding systems that cleanly factorize the codec latent space into interpretable dimensions of speaker, content, style, environment, and channel effects -- enabling precise control over each aspect and the ability to massively amplify an existing seed dataset through “latent recombination”.
Leverage latent recombination to generate synthetic audio data at previously impossible scales, unlocking joint model and data scaling paradigms for audio.  Endeavor to train multimodal speech-to-speech systems that can 1) understand any human irrespective of their demographics, state, or environment and 2) produce empathic, human-like responses that achieve conversational or task-oriented objectives.   
Design model architectures, training schemes, and inference algorithms that are adapted for hardware at the bare metal enabling cost efficient training on billion-hour datasets and powering real-time inference for hundreds of millions of concurrent conversations.
The Challenge
We are seeking researchers who:
See "unsolved" problems as opportunities to pioneer entirely new approaches
Can identify the one critical experiment that will validate or kill an idea in days, not months
Have the vision to scale successful proofs-of-concept 100x
Are obsessed with using AI to automate and amplify your own impact
If you find yourself energized rather than daunted by these expectations—if you're already thinking about five ideas to try while reading this—you might be the researcher we need. This role demands obsession with the problems, creativity in approach, and relentless drive toward elegant, scalable solutions. The technical challenges are immense, but the potential impact is transformative.

It's Important to Us That You Have
Strong mathematical foundation in statistical learning theory, particularly in areas relevant to self-supervised and multimodal learning
Deep expertise in foundation model architectures, with an understanding of how to scale training across multiple modalities
Proven ability to bridge theory and practice—someone who can both derive novel mathematical formulations and implement them efficiently
Demonstrated ability to build data pipelines that can process and curate massive datasets while maintaining quality and diversity
Track record of designing controlled experiments that isolate the impact of architectural innovations and validate theoretical insights
Experience optimizing models for real-world deployment, including knowledge of hardware constraints and efficiency techniques
History of open-source contributions or research publications that have advanced the state of the art in speech/language AI

How We Generated This Job Description
This job description was generated in two parts.  The “Opportunity”, “Role”, and “Challenge” sections were generated by a human using Claude-3.5-sonnet as a writing partner.  The objective of these sections is to clearly state the problem that Deepgram is attempting to solve, how we intend to solve it, and some guidelines to help you decide if Deepgram is right for you. Therefore, it is important that this section was articulated by a human.
  
The “It’s Important to Us” section was automatically derived from a multi-stage LLM analysis (using o1) of key foundational deep learning papers related to our research goals.  This work was completed as an experiment to test the hypothesis that traits of highly productive and impactful researchers are reflected directly in their work. The analysis focused on understanding how successful researchers approach problems, from mathematical foundations through to practical deployment. The problems Deepgram aims to solve are immensely difficult and span multiple disciplines and specialties. As such, we chose seminal papers that we believe reflect the pioneering work and exemplary human characteristics needed for success. The LLM analysis culminates in an “Ideal Researcher Profile”, which is reproduced below along with the list of foundational papers. 


Ideal Researcher Profile
An ideal researcher, as evidenced by the recurring themes across these foundational papers, excels in five key areas: (1) Statistical & Mathematical Foundations, (2) Algorithmic Innovation & Implementation, (3) Data-Driven & Scalable Systems, (4) Hardware & Systems Understanding, and (5) Rigorous Experimental Design. Below is a synthesis of how each paper highlights these qualities, with references illustrating why they matter for building robust, impactful deep learning models.


1. Statistical & Mathematical Foundations
Mastery of Core Concepts
Many papers, like Scaling Laws for Neural Language Models and Neural Discrete Representation Learning (VQ-VAE), reflect the importance of power-law analyses, derivation of novel losses, or adaptation of fundamental equations (e.g., in VQ-VAE's commitment loss or rectified flows in Scaling Rectified Flow Transformers). Such mathematical grounding clarifies why models converge or suffer collapse.
Combining Existing Theories in Novel Ways
Papers such as Moshi (combining text modeling, audio codecs, and hierarchical generative modeling) and Finite Scalar Quantization (FSQ's adaptation of classic scalar quantization to replace vector-quantized representations) show how reusing but reimagining known techniques can yield breakthroughs. Many references (e.g., the structured state-space duality in Transformers are SSMs) underscore how unifying previously separate research lines can reveal powerful algorithmic or theoretical insights.
Logical Reasoning and Assumption Testing
Across all papers—particularly in the problem statements of Whisper or Rectified Flow Transformers—the authors present assumptions (e.g., "scaling data leads to zero-shot robustness" or "straight-line noise injection improves sample efficiency") and systematically verify them with thorough empirical results. An ideal researcher similarly grounds new ideas in well-formed, testable hypotheses.

2. Algorithmic Innovation & Implementation
Creative Solutions to Known Bottlenecks
Each paper puts forth a unique algorithmic contribution—Rectified Flow Transformers redefines standard diffusion paths, FSQ proposes simpler scalar quantizations contrasted with VQ, phi-3 mini relies on curated data and blocksparse attention, and Mamba-2 merges SSM speed with attention concepts.
Turning Theory into Practice
Whether it's the direct preference optimization (DPO) for alignment in phi-3 or the residual vector quantization in SoundStream, these works show that bridging design insights with implementable prototypes is essential.
Clear Impact Through Prototypes & Open-Source
Many references (Whisper, neural discrete representation learning, Mamba-2) highlight releasing code or pretrained models, enabling the broader community to replicate and build upon new methods. This premise of collaboration fosters faster progress.

3. Data-Driven & Scalable Systems
Emphasis on Large-Scale Data and Efficient Pipelines
Papers such as Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) and BASE TTS demonstrate that collecting and processing hundreds of thousands of hours of real-world audio can unlock new capabilities in zero-shot or low-resource domains. Meanwhile, phi-3 Technical Report shows that filtering and curating data at scale (e.g., "data optimal regime") can yield high performance even in smaller models.
Strategic Use of Data for Staged Training
A recurring strategy is to vary sources of data or the order of tasks. Whisper trains on multilingual tasks, BASE TTS uses subsets/stages for pretraining on speech tokens, and phi-3 deploys multiple training phases (web data, then synthetic data). This systematic approach to data underscores how an ideal researcher designs training curricula and data filtering protocols for maximum performance.

4. Hardware & Systems Understanding
Efficient Implementations at Scale
Many works illustrate how researchers tune architectures for modern accelerators: the In-Datacenter TPU paper exemplifies domain-specific hardware design for dense matrix multiplications, while phi-3 leverages blocksparse attention and custom Triton kernels to run advanced LLMs on resource-limited devices.
Real-Time & On-Device Constraints
SoundStream shows how to compress audio in real time on a smartphone CPU, demonstrating that knowledge of hardware constraints (latency, limited memory) drives design choices. Similarly, Moshi's low-latency streaming TTS and phi-3-mini's phone-based inference highlight that an ideal researcher must adapt algorithms to resource limits while maintaining robustness.
Architectural & Optimization Details
Papers like Mamba-2 in Transformers are SSMs and the In-Datacenter TPU work show how exploiting specialized matrix decomposition, custom memory hierarchies, or quantization approaches can lead to breakthroughs in speed or energy efficiency.

5. Rigorous Experimental Design
Controlled Comparisons & Ablations
Nearly all papers—Whisper, FSQ, Mamba-2, BASE TTS—use systematic ablations to isolate the impact of individual components (e.g., ablation on vector-quantization vs. scalar quantization in FSQ, or size of codebooks in VQ-VAEs). This approach reveals which design decisions truly matter.
Multifold Evaluation Metrics
From MUSHRA listening tests (SoundStream, BASE TTS) to FID in image synthesis (Scaling Rectified Flow Transformers, FSQ) to perplexity or zero-shot generalization in language (phi-3, Scaling Laws for Neural Language Models), the works demonstrate the value of comprehensive, carefully chosen metrics.
Stress Tests & Edge Cases
Whisper's out-of-distribution speech benchmarks, SoundStream's evaluation on speech + music, or Mamba-2's performance on multi-query associative recall demonstrate the importance of specialized challenge sets. Researchers who craft or adopt rigorous benchmarks and "red-team" their models (as in phi-3 safety alignment) are better prepared to address real-world complexities.


Summary
Overall, an ideal researcher in deep learning consistently demonstrates:
A solid grounding in theoretical and statistical principles
A talent for proposing and validating new algorithmic solutions
The capacity to orchestrate data pipelines that scale and reflect real-world diversity
Awareness of hardware constraints and system-level trade-offs for efficiency
Thorough and transparent experimental practices
These qualities surface across research on speech (Whisper, BASE TTS), language modeling (Scaling Laws, phi-3), specialized hardware (TPU, Transformers are SSMs), and new representation methods (VQ-VAE, FSQ, SoundStream). By balancing these attributes—rigorous math, innovative algorithms, large-scale data engineering, hardware-savvy optimizations, and reproducible experimentation—researchers can produce impactful, trustworthy advancements in foundational deep learning.



Foundational Papers
This job description was generated through analysis of the following papers:
Robust Speech Recognition via Large-Scale Weak Supervision (arXiv:2212.04356)
Moshi: a speech-text foundation model for real-time dialogue (arXiv:2410.00037)
Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (arXiv:2403.03206)
Scaling Laws for Neural Language Models (arXiv:2001.08361)
BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data (arXiv:2402.08093)
In-Datacenter Performance Analysis of a Tensor Processing Unit (arXiv:1704.04760)
Neural Discrete Representation Learning (arXiv:1711.00937)
SoundStream: An End-to-End Neural Audio Codec (arXiv:2107.03312)
Finite Scalar Quantization: VQ-VAE Made Simple (arXiv:2309.15505)
Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone (arXiv:2404.14219)
Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (arXiv:2405.21060)
Backed by prominent investors including Y Combinator, Madrona, Tiger Global, Wing VC and NVIDIA, Deepgram has raised over $85 million in total funding. If you're looking to work on cutting-edge technology and make a significant impact in the AI industry, we'd love to hear from you!
Deepgram is an equal opportunity employer. We want all voices and perspectives represented in our workforce. We are a curious bunch focused on collaboration and doing the right thing. We put our customers first, grow together and move quickly. We do not discriminate on the basis of race, religion, color, national origin, gender, sexual orientation, gender identity or expression, age, marital status, veteran status, disability status, pregnancy, parental status, genetic information, political affiliation, or any other status protected by the laws or regulations in the locations where we operate.
