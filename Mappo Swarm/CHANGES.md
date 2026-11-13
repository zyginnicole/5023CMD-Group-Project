See README.md "Changes in v3". Untested in the authoring sandbox (no PyTorch available): mappo.update, train.py,
experiments.py, deploy.py, viva_inspect.py. Tested: tests/test_core.py (GAE with torch stubbed), plan card and
approval gate with the rule-based controller, collision attribution. Run `python -m unittest discover -s tests -v`
and a 5-update smoke run (`python src/train.py --updates 5 --run_dir runs/smoke`) first.
