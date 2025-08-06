print("Loaded druglike plugin")

import numpy as np
import math
import os
import csv
from dataclasses import dataclass
from .add_tag import add_tag
from .scoring_utils import QED, LogP, SA_score, Docking 
from reinvent_plugins.components.component_results import ComponentResults

@add_tag("__component")
class Druglike:
    def __init__(self, params):
        self.weight = [1.0]  # Can be made configurable via params
        self.params = params

    def __call__(self, smiles, valid_mask, **kwargs):
        valid_smiles = [s for s, v in zip(smiles, valid_mask) if v]
        if not valid_smiles:
            return self._empty_result(len(smiles))

        qed_scores = QED(valid_smiles)
        logp_scores = LogP(valid_smiles)
        sa_scores = SA_score(valid_smiles)
        affinities, seed = Docking(valid_smiles)

        # Scale QED [0,1] to [0,10]
        qed_scaled = np.array(qed_scores) * 10

        # Affinity [0,10] assumed to be already scaled
        affinity_scaled = np.array(affinities)

        # LogP rewards (ideal between 1.5–2.7)
        logp_scaled = []
        for lp in logp_scores:
            if 1.5 <= lp <= 2.7:
                logp_scaled.append(10)
            else:
                dist = min(abs(lp - 1.5), abs(lp - 2.7))
                penalty = 2 if dist < 0.5 else int(dist // 0.5) * 3
                logp_scaled.append(max(10 - penalty, 0))
        logp_scaled = np.array(logp_scaled)

        # SA reward
        sa_scaled = []
        for sa in sa_scores:
            if not sa or 1 <= sa <= 2.5:
                sa_scaled.append(10.0)
            elif 2.5 < sa <= 3:
                sa_scaled.append(9.0)
            else:
                sa_scaled.append(max(10 - 2 * math.ceil(sa - 3), 0.0))
        sa_scaled = np.array(sa_scaled)

        rewards = np.array([
            0.0 if aff == 0 else (0.5 * ((q + l + s) / 3.0) + 0.5 * aff)
            for q, l, s, aff in zip(qed_scaled, logp_scaled, sa_scaled, affinity_scaled)
        ], dtype=np.float32)

        full_rewards = np.zeros(len(smiles), dtype=np.float32)
        full_rewards[valid_mask] = rewards

        # print("scores: ", full_rewards)

        scores=[full_rewards]

        metadata = {
            "component_type": "druglike",
            "transformed_scores": [full_rewards],
            "raw_scores": [full_rewards],
            "weight": self.weight,
        }

        return ComponentResults(scores=scores)
    
    def _empty_result(self, n):
        metadata = {
            "component_type": "druglike",
            "transformed_scores": [np.zeros(n)],
            "raw_scores": [np.zeros(n)],
            "weight": self.weight,
        }

        return ComponentResults(scores=[np.zeros(n)])
