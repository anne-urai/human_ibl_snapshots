# This code is for the Human IBL decision-making task
- Task code at https://github.com/cami-uche-enwe/human_IBL_cursor
- Instructions [here](https://docs.google.com/document/d/1C6Kt_tYg0wLJQ1GE0N0mQVeitvk-i0vjs0vuYjYIJsQ/edit)

### Install
Get yourself an environment with MNE
- follow [these instructions](https://mne.tools/stable/install/manual_install.html#manual-install)

- install [psychofit](https://pypi.org/project/Psychofit/) for psychometric function fitting, using `pip install psychofit`.
- install `pip install audio-extract`
- install `pip install moviepy`

### Run
```python
snapshot_behavior.py # generates behavioral figures for all PsychoPy .csv files in the _data_ folder
snapshot_pupil.py # generates pupil figures for all Eyelink .asc files in the _data/human_pupil_ folder
```

---

Contact Anne Urai, Leiden University, 2024
a.e.urai@fsw.leidenuniv.nl

