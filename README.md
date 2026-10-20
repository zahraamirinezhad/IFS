# IFS for Visibility-Constrained N-Agent Ad Hoc Teamwork

**An IFS-inspired multi-agent reinforcement learning implementation integrated into the POAM/NAHT framework, with explicit random visibility masking to study coordination under restricted inter-agent observability.**

This project explores how an Information Flow Structure (IFS) can support cooperation between controlled agents and unfamiliar teammates in **N-Agent Ad Hoc Teamwork (NAHT)** settings. It builds on the NAHT/EPyMARL research codebase and adds IFS-specific communication, information-fusion, information-separation, and visibility-mask components.

> **Implementation scope:** This repository is a research implementation/adaptation, not a verified reproduction of every detail of the original IFS paper. The code currently uses a PPO-style actor–critic learner in the POAM/NAHT framework, while the IFS paper describes a value-based DDQN objective. See [Implementation fidelity and known gaps](#implementation-fidelity-and-known-gaps) before interpreting results or claiming paper-level reproduction.

## Highlights

- **IFS-inspired architecture** with Communication, Information Fusion, and Information Separation modules.
- **Visibility-constrained information fusion** using a per-timestep, pairwise agent visibility mask.
- **Ad hoc teamwork setup** with trainable/controlled agents and configurable uncontrolled teammates.
- **Auxiliary reconstruction objective** for learning fused representations of other agents' observation–action records.
- **PPO-style policy optimization** within the existing NAHT training framework.
- **Config-driven experiments** for StarCraft II / SMAC scenarios and Multi-Agent Particle Environments (MPE).
- **Experiment logging and checkpoints** through the inherited Sacred/TensorBoard workflow.

## Table of Contents

- [Research Motivation](#research-motivation)
- [Papers and Attribution](#papers-and-attribution)
- [Method Overview](#method-overview)
- [Visibility Constraints](#visibility-constraints)
- [Implementation Details](#implementation-details)
- [Environments and Configurations](#environments-and-configurations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running Experiments](#running-experiments)
- [Evaluation Protocol](#evaluation-protocol)
- [Implementation Fidelity and Known Gaps](#implementation-fidelity-and-known-gaps)
- [Results and Visualizations](#results-and-visualizations)
- [Reproducibility](#reproducibility)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Research Motivation

### N-Agent Ad Hoc Teamwork

In standard cooperative multi-agent reinforcement learning, all agents are typically trained together under a shared learning algorithm. In conventional ad hoc teamwork, a single controlled agent must cooperate with teammates whose policies may be unknown. **N-Agent Ad Hoc Teamwork (NAHT)** generalizes these settings: multiple controlled agents must cooperate with a dynamically changing team that can also contain uncontrolled or unfamiliar agents.

This introduces challenges such as:

- Partial observability of the environment and teammates
- Changing team composition and teammate policies
- Uncertainty about how uncontrolled agents make decisions
- Limited information exchange between agents
- The need to generalize beyond a single fixed team

The POAM method addresses NAHT by learning representations of teammate behavior. The IFS paper approaches the problem from an information-flow perspective, emphasizing communication between controlled agents and more effective fusion of variable-size teammate information.

### Goal of this project

The goal of this project is to integrate IFS-inspired information processing into the POAM/NAHT codebase and investigate how **restricted visibility between agents** affects information fusion and cooperative behavior.

The visibility mask limits which other agents' records are available to the information-fusion module. This provides a basis for controlled experiments on information availability rather than assuming that every agent can always observe every teammate.

## Papers and Attribution

### 1. IFS: Information Flow Structure for Multi-agent Ad Hoc System

**Primary algorithmic reference**

- **Authors:** Yanqing Fu, Chenrun Wang, Chao Huang, and Zhuping Wang
- **arXiv preprint:** October 25, 2025
- **arXiv ID:** `2510.22320`
- **DOI:** [10.48550/arXiv.2510.22320](https://doi.org/10.48550/arXiv.2510.22320)
- **Links:** [arXiv abstract](https://arxiv.org/abs/2510.22320) · [PDF](https://arxiv.org/pdf/2510.22320)

**Problem addressed:** Multi-agent ad hoc systems must cooperate with both known and unknown teammates in dynamic environments where agents, behaviors, and available information can vary.

**Main ideas in the paper:**

- **Communication Module (CM):** Generates communication information from a controlled agent's internal state to help other controlled agents infer its intentions.
- **Information Fusion Module (IFM):** Converts variable-length information about other agents into a fixed-size representation.
- **Information Separation Module (ISM):** Reconstructs or separates agent-specific information from the fused representation, providing an auxiliary learning signal.
- **Communication protocol for controlled agents:** Supports information exchange among controlled teammates without assuming that uncontrolled agents share the same communication protocol.
- **Generalization to changing team composition:** Aims to reduce dependence on a fixed maximum number and ordering of agents.

The original paper reports experiments in StarCraft II and describes a **value-based DDQN-style objective**. The implementation in this repository adapts IFS-inspired modules to the existing POAM/NAHT codebase; implementation differences are listed below.

### 2. N-Agent Ad Hoc Teamwork

**Foundational paper for the environment and POAM baseline**

- **Authors:** Caroline Wang, Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone
- **Venue:** Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Main Conference Track
- **DOI:** [10.52202/079017-3551](https://doi.org/10.52202/079017-3551)
- **Links:** [Official NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cabf611498431ad89a85ace75f790d93-Abstract-Conference.html) · [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cabf611498431ad89a85ace75f790d93-Paper-Conference.pdf) · [Official code repository](https://github.com/carolinewang01/naht)

**Main contribution:** The paper formalizes NAHT and proposes **Policy Optimization with Agent Modelling (POAM)**, a policy-gradient MARL approach that learns representations of teammate behavior to improve cooperation with diverse and previously unseen teammates.

**Connection to this repository:** This project uses the NAHT/EPyMARL code structure, open-team training configuration patterns, agent-loading utilities, and experiment infrastructure. POAM is the conceptual and codebase baseline for the integration; it is not the same algorithm as IFS.

### Relationship between the two papers and this implementation

| Element | Source / role |
| --- | --- |
| NAHT problem formulation and open-team setting | Wang et al., NeurIPS 2024 |
| POAM baseline and inherited training infrastructure | Wang et al., NeurIPS 2024 codebase |
| Communication, information fusion, and information separation concepts | Fu et al., arXiv:2510.22320 |
| Random pairwise visibility masking | This repository's implementation extension |
| PPO-style policy optimization | Existing learner framework adapted for this implementation; differs from the original IFS paper's DDQN-style objective |
| Performance claims | Must be established through experiments in this repository; not inferred from the cited papers |

## Method Overview

The code implements three main IFS-inspired modules and connects them to the NAHT agent and learner pipeline.

```text
Per-agent local observations + previous actions
                       │
                       ▼
        Pairwise observation–action records
                       │
                       ▼
              Visibility mask
       (which teammate records are available)
                       │
                       ▼
        Information Fusion Module (IFM)
       masked features → attention weights
                       │
                       ▼
            Fused representation eᵢ
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
     Recurrent state bᵢ   Information Separation
             │             Module (ISM)
             ▼                   │
   Communication Module          ▼
        message dᵢ         reconstructed records
             │                   │
             ▼                   ▼
  Communication output      reconstruction loss

Policy / value learning is handled by the adapted
NAHT actor–critic training pipeline.
```

This is a conceptual overview of the module implementation. In the current agent forward path, the generated communication vector is not explicitly routed into other agents' policy inputs; see [Implementation fidelity and known gaps](#implementation-fidelity-and-known-gaps).

### Communication Module (CM)

The communication module projects the agent's recurrent internal state through a fully connected layer, ReLU activation, and LayerNorm to produce a message representation:

\[
d_i = \operatorname{LayerNorm}(\operatorname{ReLU}(W_c b_i + b_c)).
\]

In the supplied code, this projection is implemented in `CommunicationModule` in `src/components/ifs/ifs.py`.

### Information Fusion Module (IFM)

The fusion module processes a variable number of pairwise agent records. It applies a learned feature projection, scores the records using a trainable vector, normalizes the scores with softmax, and computes a weighted representation. If a visibility mask is supplied, invisible records are zeroed and their attention scores are masked.

This produces a fixed-size representation even when the number of visible teammate records changes.

### Information Separation Module (ISM)

The separation module maps the fused representation back into a set of agent-record slots. During training, the learner compares reconstructed records with the corresponding input records using a mean-squared reconstruction objective over the visible entries.

The auxiliary reconstruction loss is logged as `ifs_recon_loss`.

### Policy and value learning

The agent and critic are integrated with the inherited `AgentOwnedMAC` / `OpenTrainMAC` controller framework. The current `IFSLearner` uses PPO-style clipped policy updates, advantage estimation, and a value critic. This is an implementation adaptation and should not be described as the exact DDQN training objective from the IFS paper.

## Visibility Constraints

Visibility is represented as a pairwise mask over agents. In the current implementation, the mask is generated by `extract_visibility_mask` in `src/components/ifs/ifs.py`.

### Current default behavior

- **Random visibility probability:** `availability_prob = 0.7`
- **Diagonal:** Self-links are excluded; the model processes records of other agents.
- **Temporal sampling:** A mask is generated for the observation batch passed to the agent/learner.
- **Pairwise masking:** The mask is aligned with each controlled agent's list of other-agent records before being passed to the fusion module.
- **Optional utility arguments:** The mask helper supports distance-based filtering, symmetric visibility, and random occlusion, but these options are not exposed by the current IFS agent call and therefore remain at their function defaults unless the code is changed.

The mask can be interpreted as:

\[
M_{ij} =
\begin{cases}
1, & \text{if agent } i \text{ can use agent } j\text{'s record},\\
0, & \text{otherwise}.
\end{cases}
\]

For the current random setting, visibility edges are sampled independently with the configured availability probability (except for the excluded diagonal). This is a **random visibility constraint**, not a verified physical sensor model or geometric line-of-sight model.

### Recommended visibility experiments

To study visibility rigorously, expose these parameters in YAML configuration and compare:

| Experiment | Description |
| --- | --- |
| Full visibility | Every other-agent record is available |
| High visibility | `availability_prob` close to 1 |
| Medium visibility | `availability_prob = 0.7` (current default) |
| Low visibility | Lower visibility probabilities, such as 0.3 or 0.5 |
| Symmetric visibility | Visibility from \(i\) to \(j\) implies visibility from \(j\) to \(i\) |
| Distance-based visibility | Keep only agents within a specified distance, when the observation features contain meaningful positions |
| Random occlusion | Independently remove additional visibility edges |
| POAM baseline | Run POAM under a matched environment and evaluation protocol |

These are suggested ablations, not reported results. Make sure each mode is actually passed into the agent and record the exact configuration, seed, and mask-resampling policy.

## Implementation Details

| Component | File |
| --- | --- |
| IFS modules and visibility utilities | `src/components/ifs/ifs.py` |
| IFS policy agent | `src/modules/agents/ifs_agent.py` |
| IFS value critic | `src/modules/critics/ifs.py` |
| IFS learner | `src/learners/ifs_learner.py` |
| POAM baseline learner | `src/learners/poam_learner.py` |
| MPE IFS configuration | `src/config/algs/mpe/ifs.yaml` |
| SC2 IFS configuration | `src/config/algs/sc2/ifs.yaml` |
| Open-team experiment configurations | `src/config/open/ifs/` |
| Main experiment entry point | `src/main.py` |
| Experiment runner | `src/run.py` |
| Results directory configuration | `src/config/user_info.yaml` |

### Main MPE IFS hyperparameters

The supplied `src/config/algs/mpe/ifs.yaml` includes these settings:

| Parameter | Configured value |
| --- | ---: |
| `availability_prob` | 0.7 |
| `message_dim` | 64 |
| `fused_dim` | 128 |
| `n_train_slots` | 5 |
| `embed_dim` | 128 |
| `lr` | 0.0003 |
| `critic_lr` | 0.0003 |
| `ifs_lr` | 0.0003 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `eps_clip` | 0.2 |
| `epochs` | 5 |
| `n_minibatch` | 2 |
| `entropy_coef` | 0.01 |
| `grad_norm_clip` | 0.5 |
| `batch_size_run` | 8 |
| `batch_size` | 256 |

These are configuration values, not optimized or independently validated hyperparameters. The SC2 IFS configuration file in the supplied archive is empty and must be populated before using `--alg-config=sc2/ifs`.

## Environments and Configurations

The archive contains open-team configuration templates for:

### StarCraft II / SMAC

- `5m_vs_6m`
- `8m_vs_9m`
- `10m_vs_11m`
- `3s_vs_5z`

The associated files are under `src/config/open/ifs/`. They configure controlled agents and load checkpoints for uncontrolled teammates. Those checkpoint paths are machine- and experiment-specific and must be updated to point to valid local models.

**Important:** The supplied `src/config/algs/sc2/ifs.yaml` is a zero-byte file. The SC2 configuration cannot be treated as ready to run until this file is filled with the intended IFS settings, based on the working MPE IFS configuration and the required SC2-specific values.

### Multi-Agent Particle Environments (MPE)

The archive includes an open-team Predator–Prey configuration, `src/config/open/ifs/open_train_pp_ifs.yaml`, and an IFS algorithm configuration at `src/config/algs/mpe/ifs.yaml`. This is the most complete IFS configuration pair in the supplied archive, but it still depends on the custom MPE environment and valid pretrained teammate checkpoints.

## Repository Structure

```text
.
├── requirements.txt
├── env_requirements.txt
├── install_sc2.sh
├── train_agent.sh
├── train_naht.sh
├── run_interactive.sh
├── run_test.sh
└── src/
    ├── main.py
    ├── run.py
    ├── components/
    │   ├── episode_buffer.py
    │   └── ifs/
    │       └── ifs.py
    ├── config/
    │   ├── algs/
    │   │   ├── mpe/
    │   │   │   └── ifs.yaml
    │   │   └── sc2/
    │   │       └── ifs.yaml
    │   ├── envs/
    │   └── open/
    │       └── ifs/
    ├── controllers/
    ├── envs/
    ├── learners/
    │   └── ifs_learner.py
    ├── modules/
    │   ├── agents/
    │   │   └── ifs_agent.py
    │   └── critics/
    │       └── ifs.py
    ├── runners/
    └── utils/
```

## Installation

The codebase inherits a research-oriented PyMARL/EPyMARL environment and includes legacy dependencies. A Linux machine is recommended for StarCraft II / SMAC experiments.

### Requirements

- Python environment compatible with the dependency versions
- PyTorch (GPU/CUDA is optional but useful for training)
- StarCraft II and SMAC maps for SC2 experiments
- Custom MPE and matrix-game packages listed in `env_requirements.txt`
- Sacred and TensorBoard for experiment tracking

The supplied requirements include older packages such as `gym==0.23.1`, `pysc2==3.0.0`, and `PyYAML==5.3.1`. These can conflict with modern Python environments. Use a dedicated environment and pin the versions that you verify.

### 1. Clone the repository

After publishing your own repository, replace the placeholder URL:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Create a virtual environment

A Conda environment is recommended. The upstream NAHT repository documents Python 3.10 and PyTorch 2.5 for its verified setup, while this archive also contains legacy dependencies; verify compatibility before selecting the final versions.

```bash
conda create -n ifs-naht python=3.10
conda activate ifs-naht
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r env_requirements.txt
```

If dependency installation fails, resolve version conflicts for the Python, PyTorch, SMAC, PySC2, and custom MPE packages. Record the exact working versions in a lock file or environment specification.

### 4. Install StarCraft II / SMAC (for SC2 experiments)

Review `install_sc2.sh` and the upstream [NAHT repository instructions](https://github.com/carolinewang01/naht) before running installation scripts. The supplied script assumes a particular directory layout and downloads legacy game/map assets; verify its paths and URLs for your machine.

## Running Experiments

All experiments are launched through `src/main.py` and configured using YAML files.

### MPE: IFS open-team experiment

The archive contains a matching MPE algorithm config and open-team config. After updating teammate checkpoint paths and confirming the custom MPE environment is installed, use:

```bash
python src/main.py \
  --config=open/ifs/open_train_pp_ifs \
  --alg-config=mpe/ifs \
  --env-config=mpe \
  with t_max=3000000 \
  env_args.key="mpe:PredatorPrey-v0" \
  env_args.time_limit=100 \
  env_args.pretrained_wrapper="PretrainedTag" \
  --seed=112358
```

This command follows the configuration pattern in `run_test.sh`. It has not been executed as part of preparing this README; dependency and checkpoint setup may be required first.

### SC2: IFS open-team experiment

The repository includes SC2 open-team templates, for example `src/config/open/ifs/open_train_5v6_ifs.yaml`, but the corresponding `src/config/algs/sc2/ifs.yaml` is empty in the supplied archive. Populate and validate that algorithm config before attempting a SC2 IFS run.

Once the SC2 IFS config is complete and teammate checkpoint paths are valid, the intended command pattern is:

```bash
python src/main.py \
  --env-config=sc2 \
  --config=open/ifs/open_train_5v6_ifs \
  --alg-config=sc2/ifs \
  with env_args.map_name=5m_vs_6m \
  --seed=112358
```

### POAM baseline

For reference, the upstream NAHT codebase documents this POAM command for StarCraft II:

```bash
python src/main.py \
  --env-config=sc2 \
  --config=open/open_train_5v6 \
  --alg-config=sc2/poam \
  with env_args.map_name=5m_vs_6m \
  --seed=112358
```

Use the same environment, team composition, evaluation episodes, and seed set when comparing POAM with the IFS adaptation.

## Evaluation Protocol

For a credible research comparison, evaluate IFS and POAM under matched conditions. Record:

- Environment and map (for example, `5m_vs_6m` or Predator–Prey)
- Number of controlled and uncontrolled agents
- Uncontrolled teammate policy types and checkpoint identifiers
- Visibility probability and mask type
- Whether visibility is resampled at every forward pass, timestep, or episode
- Training budget in environment timesteps
- Random seeds and number of independent runs
- Mean evaluation return and standard deviation or confidence interval
- Win/success rate where the environment supports it
- Wall-clock training time and compute setup
- Model-selection and checkpoint-selection protocol

Useful ablations include full visibility versus restricted visibility, multiple visibility probabilities, and IFS-inspired modules versus the POAM baseline. Report only metrics generated from actual runs; paper results are not evidence of results from this repository.

## Implementation Fidelity and Known Gaps

A professional research repository should make implementation differences explicit. The following details were identified in the supplied archive:

1. **Training objective differs from the IFS paper.** The original IFS paper describes a value-based DDQN-style objective. `src/learners/ifs_learner.py` instead implements PPO-style clipped policy optimization and value learning in the existing NAHT framework. Describe this as an IFS-inspired adaptation unless you implement and validate the original objective.
2. **The communication vector is not routed through the policy input.** `IFSArchitectureWithConstraints.forward()` computes `d_i` through the Communication Module, but `IFSAgent.forward()` currently uses `e_i` and the recurrent state while not explicitly passing `d_i` to other agents. As a result, the communication projection exists in the code but the full inter-agent message-passing path described in the paper is not visibly wired into the policy forward pass.
3. **Visibility is currently random, not geometric.** The agent calls `extract_visibility_mask()` with `availability_prob` only. The helper supports distance filtering, symmetry, and occlusion, but those options are not configured by the current call.
4. **The SC2 IFS config is empty.** `src/config/algs/sc2/ifs.yaml` has no contents in the uploaded archive. Fill and validate it before running SC2 IFS experiments.
5. **Uncontrolled teammate checkpoints are external.** The open-team YAML files contain checkpoint paths that may not exist on another machine. Update `src/config/user_info.yaml` and the relevant configuration files for your local setup.
6. **No result assets were included.** The archive contains no GIFs, videos, plots, or benchmark-result figures. Add them after running and validating experiments.
7. **The test script contains commented examples.** `run_test.sh` is primarily a collection of commented command templates, not an automated test suite.

These points do not prevent the repository from being presented as a meaningful implementation project. They clarify what has been implemented and identify the next steps required for a closer reproduction of the paper.

## Results and Visualizations

No rollout GIFs, training curves, or evaluation tables were included in the uploaded archive. Add visuals after generating them, for example:

```text
results/
├── ifs_visibility_0.3.png
├── ifs_visibility_0.7.png
├── ifs_full_visibility.png
├── ifs_vs_poam_returns.png
└── predator_prey_rollout.gif
```

Then embed them using repository-relative paths:

```markdown
![IFS vs. POAM evaluation return](results/ifs_vs_poam_returns.png)

![IFS rollout](results/predator_prey_rollout.gif)
```

Do not include placeholder metrics or claim improvements until the corresponding experiments have been run.

## Reproducibility

- Set and record random seeds for Python, NumPy, and PyTorch.
- Pin dependency versions after verifying a working installation.
- Save the full resolved configuration with each experiment.
- Use multiple independent seeds and report variability.
- Record the exact teammate checkpoint files and team composition.
- Make visibility-mask generation configurable and document when masks are resampled.
- Keep the upstream license and attribution when modifying or redistributing the base repository.
- Separate results from the original papers from results produced by this implementation.

## Acknowledgements

This project builds on the NAHT/EPyMARL codebase and its open-team training and evaluation infrastructure.

Thanks to the authors of the research works that motivate this project:

- Yanqing Fu, Chenrun Wang, Chao Huang, and Zhuping Wang, *IFS: Information Flow Structure for Multi-agent Ad Hoc System*.
- Caroline Wang, Arrasy Rahman, Ishan Durugkar, Elad Liebman, and Peter Stone, *N-Agent Ad Hoc Teamwork*.

## Citation

**IFS: Information Flow Structure for Multi-agent Ad Hoc System**

```bibtex
@misc{fu2025ifs,
  title         = {IFS: Information Flow Structure for Multi-agent Ad Hoc System},
  author        = {Fu, Yanqing and Wang, Chenrun and Huang, Chao and Wang, Zhuping},
  year          = {2025},
  eprint        = {2510.22320},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA},
  doi           = {10.48550/arXiv.2510.22320},
  url           = {https://arxiv.org/abs/2510.22320}
}
```

**N-Agent Ad Hoc Teamwork**

```bibtex
@inproceedings{wang2024naht,
  title     = {N-Agent Ad Hoc Teamwork},
  author    = {Wang, Caroline and Rahman, Arrasy and Durugkar, Ishan and Liebman, Elad and Stone, Peter},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024},
  doi       = {10.52202/079017-3551},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/cabf611498431ad89a85ace75f790d93-Abstract-Conference.html}
}
```
