# MA<sup>2</sup>E : Addressing Partial Observability in Multi-Agent Reinforcement Learning with Masked Auto-Encoder
<a href="[https://arxiv.org/abs/2405.19806](https://openreview.net/forum?id=klpdEThT8q&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))"><img src="https://img.shields.io/badge/Paper-OpenReview-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a>

## Description 
This is the official code repository for the paper ["MA<sup>2</sup>E : Addressing Partial Observability in Multi-Agent Reinforcement Learning with Masked Auto-Encoder"](https://openreview.net/forum?id=klpdEThT8q&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)) accepted by ICLR 2025.
MA<sup>2</sup> is a novel approach to addressing partial observability in multi-agent reinforcement learning (MARL). MA<sup>2</sup>E enables agents to infer global information solely from local observations by leveraging masked auto-encoders (MAE), eliminating the need for explicit communication.

## Installation

## Running Script
```
* bash run.sh config_name env_config_name map_name_list (arg_list threads_num gpu_list experinments_num)
* Example : bash run.sh qmix sc2 3s_vs_5z use_MT=True 3 0 3 => Run qmix+MA2E in SMAC 3s_vs_5z scenario
```
'use_MT' arg means executing the model taht plugs in MA2E into the baseline algorithm. 
