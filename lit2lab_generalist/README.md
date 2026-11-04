# Language-Model Training and Distillation

This module contains the language-model workflows for **[Lit2Lab Reactivity](../README.md)**: literature-reaction preparation, teacher annotation requests, reaction-type specialist training, generalist distillation, and offline inference.

## Workflow guide


| Component                   | Description                                                                                   | Guide                                                                |
| --------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Acid–amine specialist       | Reaction preparation, teacher annotation, label-blind GPT filtering, SFT, GRPO, and inference | [Acid–amine workflow](acidamine/README.md)                           |
| Suzuki–Miyaura specialist   | Reaction preparation, teacher annotation requests, SFT, GRPO, and inference                   | [Suzuki workflow](suzuki/README.md)                                  |
| Buchwald–Hartwig specialist | SFT, GRPO, offline inference, and evaluation                                                  | [Buchwald workflow](buchwald/README.md)                              |
| Generalist distillation     | Select correct specialist outputs, merge reaction types, and train the generalist             | [Multi-reaction distillation](multi_reaction_distillation/README.md) |
| Shared utilities            | Reaction descriptions, annotation prompts, batch requests, rewards, and inference helpers     | [Shared utilities](common/README.md)                                 |




## Workflow

The workflow follows six steps:

1. **Prepare literature reactions.** Standardize reaction inputs, assign binary yield classes, and generate atom mappings and structural descriptions for teacher annotation.
2. **Construct rationale supervision.** Generate teacher rationales conditioned on the recorded reaction and its observed outcome.
3. **Train reaction-type specialists.** Apply SFT followed by Group Relative Policy Optimization (GRPO), initializing GRPO from the corresponding SFT checkpoint.
4. **Generate and filter specialist outputs.** Run each specialist on reaction inputs without observed outcomes, retaining rationales and predictions whose final labels match the recorded outcomes.
5. **Train the generalist.** Merge the retained examples across reaction types and fine-tune a pretrained base model by SFT.
6. **Predict target reactions.** Use the generalist to generate pseudo labels for target reactions. These predictions are used for BNN warm-prior construction and active learning in [warm_prior/](../warm_prior/README.md).



## Setup and usage

Run the workflow commands from `lit2lab_generalist/`. See the component guides above for environment setup and execution commands.