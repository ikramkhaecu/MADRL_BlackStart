# PI-MADRL Black-Start Restoration — Repository v2

Simulation code for *“Physics-Informed Multi-Agent DRL Framework for Resilient System Restoration in Inverter-dominated Hybrid Power Plants.”*

This repository provides the complete simulation framework for physics-informed multi-agent deep reinforcement learning (PI-MADRL) for resilient black-start and system-restoration studies in inverter-dominated hybrid power plants.

Repository v2 provides the updated implementation with the electrical modelling, restoration environment, physics-informed learning components, baselines, experiment pipeline, testing framework, and result-generation tools used for the study.

**The complete source code, simulation framework, configuration files, and supporting data required to reproduce the study are available in repository version 1.4.**

## Layout

| Path                              | Purpose                                                                                                                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `configs/default.yaml`            | Central configuration containing simulation, network, environment, training, and experiment parameters                                                                                                                         |
| `sim/networks.py`                 | Modified IEEE 14-, 39-, and 118-bus network models based on pandapower `case14`, `case39`, and `case118`                                                                                                                       |
| `sim/acpf.py`                     | Droop-based islanded AC power flow without a slack bus, including P–f and Q–V droop, unit limits, islands, and sensitivity calculations                                                                                        |
| `sim/frequency.py`                | Aggregated VSM and droop frequency-response model for RoCoF, frequency nadir, and admissible load-step calculations                                                                                                            |
| `sim/blackstart_env.py`           | Dec-POMDP restoration environment with five functional agents: switch, PV, WP, ESS, and load; includes action masking, look-ahead shielding, protection, CLPU, synchronisation checks, SoC constraints, and reward formulation |
| `sim/nets.py`, `sim/algos.py`     | DAN encoder, actor and critic networks with physics-informed heads, QMIX mixer, and DQN, IQL, VDN, QMIX, PPO, MAPPO, and PI-MAPPO algorithms                                                                                   |
| `sim/train.py`                    | Training, validation, and testing framework, including the component matrix and S1–S5 experimental protocols                                                                                                                   |
| `sim/baselines.py`                | Random with masking, greedy operator, beam-search, and relaxed-bound baselines                                                                                                                                                 |
| `sim/run_experiments.py`          | Main experiment pipeline for training and evaluating the algorithms                                                                                                                                                            |
| `sim/make_tables.py`              | Generates LaTeX result tables and summary claims from experiment outputs                                                                                                                                                       |
| `figs/make_comparison_figures.py` | Generates comparison and performance figures                                                                                                                                                                                   |
| `tests/`                          | Automated tests covering electrical physics, frequency models, network data, environment behaviour, PBRS, masking, and the experimental pipeline                                                                               |
| `docs/`                           | Documentation covering parameters, project notes, publication/reproduction instructions, and related project documentation                                                                                                     |
| `sim/results/legacy_v1/`          | Archived results from the previous implementation, retained for historical reference                                                                                                                                           |

## Install and Check

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Run the complete test suite:

```bash
python -m pytest -q
```

Inspect parameters that require confirmation:

```bash
python -m sim.config --provisional
```

Display the network descriptors:

```bash
python -m sim.networks
```

## Reproduce the Experiments

### Pipeline Check

A reduced experiment can be used to verify the complete simulation and result-generation pipeline:

```bash
python -m sim.run_experiments --quick --out sim/results_quick
```

### Full Experimental Matrix

The complete experimental matrix includes the baselines, benchmark experiments, S1–S5 studies, stress tests, fair comparisons, and ablation experiments:

```bash
python -m sim.run_experiments --seeds 10 --episodes 3000 --jobs 16
```

Generate the LaTeX tables and summary claims:

```bash
python -m sim.make_tables
```

Generate the comparison figures:

```bash
python figs/make_comparison_figures.py
```

### Single Training Run

A single PI-MAPPO experiment can be executed using:

```bash
python -m sim.train --algo pi-mappo --system 14 --seed 0 --episodes 1000
```

The experiment pipeline supports selecting specific systems and experimental subsets using the corresponding `--systems` and `--parts` options.

## Experimental Protocol

The framework supports experiments across modified IEEE 14-, 39-, and 118-bus systems and evaluates multiple multi-agent reinforcement-learning approaches and non-learning baselines.

The experimental framework includes:

* Random-action and action-masked random baselines
* Greedy operator baseline
* Beam-search baseline
* Relaxed-bound reference
* DQN
* IQL
* VDN
* QMIX
* PPO
* MAPPO
* PI-MAPPO
* S1–S5 experimental scenarios
* Stress-testing experiments
* Fair algorithm comparisons
* Ablation studies

Training, validation, and test scenarios use separate random seeds. Model selection is performed using validation scenarios, while the reported performance is obtained from the test scenarios.

## Physics-Informed Restoration Framework

The PI-MADRL framework incorporates electrical-system physics directly into the restoration decision-making process.

The implementation includes:

* Islanded AC power-flow modelling
* P–f and Q–V droop characteristics
* Virtual synchronous-machine and droop-based frequency response
* RoCoF and frequency-nadir constraints
* State-of-charge constraints for energy-storage systems
* Generator and inverter operating limits
* Network switching constraints
* Action masking
* Look-ahead safety shielding
* Protection constraints
* Cold-load pickup (CLPU)
* Synchronisation checks
* Physics-informed network representations
* Physics-informed critic components
* Potential-based reward shaping (PBRS)
* Decentralised multi-agent policies with centralised training

Five functional agents represent the principal restoration-control functions:

1. **Switch agent**
2. **PV agent**
3. **Wind-power (WP) agent**
4. **Energy-storage-system (ESS) agent**
5. **Load agent**

## Reporting and Evaluation

The experiment pipeline distinguishes between the training signal and the environment's actual return.

The reported episode return is the **unshaped environment return (`ret`)**. The shaped training signal is recorded separately as **`ret_shaped`**.

The evaluation framework also records the relevant physics-informed components, including:

* Physics-informed state/features
* Action masking
* Potential-based reward shaping
* Physics-informed loss components
* DAN representations

An episode in which the electrical island collapses is assigned:

```text
LRR = 0
```

Result files contain metadata identifying the corresponding experiment and run configuration. This metadata is also propagated to generated figures and tables to distinguish different experiment versions and configurations.

## Computational Requirements

The computational cost depends on the network size and training configuration.

Approximate PI-MAPPO execution time for a single-core run is:

| System       | Approx. time per episode |
| ------------ | -----------------------: |
| IEEE 14-bus  |                  ≈ 0.2 s |
| IEEE 39-bus  |                  ≈ 0.4 s |
| IEEE 118-bus |                    ≈ 2 s |

Experiments are independent and can therefore be executed in parallel using the `--jobs` option.

For example:

```bash
python -m sim.run_experiments --seeds 10 --episodes 3000 --jobs 16
```

The number of seeds and episodes should be selected according to the experimental objective and the convergence behaviour observed in the learning curves.

## Data and Reproducibility

The repository is intended to provide the complete computational material required for reproducing the reported experiments.

**The full code and supporting data are available in repository version 1.4.**

The repository contains the simulation source code, network definitions, configuration files, training and evaluation procedures, baseline implementations, automated tests, experiment scripts, and result-processing utilities required for the study.

The archived legacy results are retained under:

```text
sim/results/legacy_v1/
```

These files are preserved for historical and reproducibility purposes and should not be treated as outputs generated by the current implementation.

## Repository Version

This repository corresponds to **PI-MADRL Black-Start Restoration — v2**.

The complete code and data associated with the project are available through **repository version 1.4**.
