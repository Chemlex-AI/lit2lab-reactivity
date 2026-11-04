# Lit2Lab Reactivity

**From literature-trained language models to data-efficient reaction exploration.**

Code and data for **[Bridging the Literature-to-Laboratory Gap with Language Models and Active Learning for Reaction Feasibility Prediction and Autonomous Exploration](https://doi.org/10.26434/chemrxiv.15011623/v1)**.

The study covers three reaction classes: **acid–amine condensation, Suzuki–Miyaura coupling, and Buchwald–Hartwig amination**.

## Repository structure

The repository is organized into two modules, with installation and usage instructions in their respective workflow READMEs.

```text
lit2lab-reactivity/
├── README.md
├── lit2lab_generalist/   # Language-model training, distillation, and inference
│   ├── README.md        # Language-model workflow guide
│   ├── acidamine/
│   ├── suzuki/
│   ├── buchwald/
│   ├── multi_reaction_distillation/
│   ├── common/
│   └── scripts/
└── warm_prior/          # HTE data, Bayesian warm priors, and active learning
    └── README.md        # Warm-prior installation and usage guide
```

| Module                                    | Scope                                                                                                                       | Documentation                                               |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **Language models**                 | Reaction preprocessing, teacher annotation, specialist SFT and GRPO, rejection-sampling distillation, and offline inference | [lit2lab_generalist/README.md](lit2lab_generalist/README.md) |
| **Warm priors and active learning** | HTE benchmarks, Generalist-derived pseudo-labels, BNN prior construction, acquisition experiments, and evaluation           | [warm_prior/README.md](warm_prior/README.md)                 |

## How the modules connect

```text
lit2lab_generalist/
Literature reactions → SFT + GRPO specialists → Generalist distillation
                                                    ↓
                                      Target-reaction pseudo-labels
                                                    ↓
warm_prior/
Bayesian warm priors → Experimental-label updates → Active learning
```

The `lit2lab_generalist/` module trains reaction-type specialists and distills their accepted rationale–label outputs into a generalist model. The generalist predicts reaction feasibility in target HTE domains without target-domain language-model training.

The `warm_prior/` module transfers Generalist predictions into BNN priors. The reliability-adjusted variant uses initial experimental labels to estimate pseudo-label sensitivity and specificity. Subsequent experimental labels update the BNN, while epistemic uncertainty guides acquisition. The generalist remains fixed during this stage.

The active-learning study uses previously measured HTE outcomes in **retrospective pool-based simulations**, evaluating label efficiency and providing a basis for deployment in automated laboratory workflows.

## Getting started

Clone the repository:

```bash
git clone https://github.com/Chemlex-AI/lit2lab-reactivity.git
cd lit2lab-reactivity
```

Use the module-specific environments and working directories:

- **Language-model training or inference:** start with [lit2lab_generalist/README.md](lit2lab_generalist/README.md) and run its workflow commands from `lit2lab_generalist/`.
- **BNN adaptation or active learning:** start with [warm_prior/README.md](warm_prior/README.md) and run its commands from `warm_prior/`.

The modules use different software stacks. Installation instructions, data formats, and execution commands are documented in their respective READMEs.

## Study settings

Reaction feasibility is defined as binary yield classification:

| Recorded yield   | Language-model label | Binary label |
| ---------------- | -------------------- | ------------ |
| Greater than 20% | `high`             | `1`        |
| At most 20%      | `low`              | `0`        |

For warm-prior construction, the study assigns pseudo-labels by majority vote across seven Generalist rollouts, using valid predictions. The Bayesian active-learning experiments compare four strategies:

| Strategy                        | Prior                                                               | Acquisition             |
| ------------------------------- | ------------------------------------------------------------------- | ----------------------- |
| Vanilla BNN                     | Base parameter prior                                                | Epistemic uncertainty   |
| Naive warm prior                | Generalist pseudo-labels treated as error-free                      | Epistemic uncertainty   |
| Reliability-adjusted warm prior | Generalist pseudo-labels with estimated sensitivity and specificity | Epistemic uncertainty   |
| Random-acquisition control      | The same reliability-adjusted prior                                 | Uniform random sampling |

Held-out test reactions are excluded from prior construction, reliability estimation, posterior updating, and acquisition. See the manuscript, Supplementary Information, and module documentation for the complete protocols.

## Data

The `lit2lab_generalist/` module includes small examples demonstrating training-data formats and workflow execution. Their provenance and scope are documented in the [language-model guide](lit2lab_generalist/README.md). The HTE data, pseudo-label assets, and active-learning inputs are documented in the [warm-prior guide](warm_prior/README.md).

## Citation

If you use this work, please cite:

```bibtex
@article{zhong2026lit2lab,
  title   = {Bridging the Literature-to-Laboratory Gap with Language Models and Active Learning for Reaction Feasibility Prediction and Autonomous Exploration},
  author  = {Zhong, Haowen and Yang, Yi and Jia, Jiekai and Yan, Xin and Xie, Yongli and Li, Yuan and Cui, Xiaofeng and Lin, Sen and Wang, Peng and Wang, Xiaoxue},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15011623/v1},
  url     = {https://doi.org/10.26434/chemrxiv.15011623/v1}
}
```
