"""fastsim -- reduced-order surrogate of the Smart Buildings thermal step.

See docs/fastsim_rom.md for the equations this reproduces and the measured
fidelity. Nothing here is used for scoring: official numbers come from the real
simulator, this only exists to make tuning and RL affordable.

Layout
------
``scen``     helper to import one simulator tree and build an environment
``extract``  pull the exact per-timestep operators out of the real solver
``record``   instrumented CtrlW rollout that logs everything the solve consumes
``rom``      reduced basis (block-Krylov + POD) and the reduced timestep
``replay``   open-loop fidelity of the ROM against a logged rollout
``bench``    per-step cost of the reduced update vs the real solve
"""
