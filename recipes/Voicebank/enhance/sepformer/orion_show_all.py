
import pprint

from orion.client import get_experiment
import sys

experiment = get_experiment(name='vctk-sepformer-hparam-search-test-12-v1')

pprint.pprint(experiment.stats)

for trial in experiment.fetch_trials():
    print(trial.id)
    print(trial.status)
    print(trial.params)
    print(trial.results)
    print()
    pprint.pprint(trial.to_dict())

# Fetches only the completed trials
for trial in experiment.fetch_trials_by_status('completed'):
    print(trial.objective)
