# AGENTS.md

This repository evaluates EEG models on various downstream applications.

Use `./.conda` environment for testing codes.

## Grill me

Always ask for clarifications on technical deatils of your implementations rather than making implicit assumptions on the code behaviours by yourself.

## Boundaries and Constraints

NEVER edit or change any file nor the file structure of any folder in `raw_roots`.
NEVER reveal local paths on public configuration files or scripts (this include tests). You should ALWAYS keep local paths, tokens in files like `.local.yaml`. For example, you should NOT include a local run directory under `assets/run` in tests.

NEVER re-run experiments that already have artifacts -- complete waste of computational resource.

AVOID writing codes in a way that destroy or overwrite artifacts of already completed runs, unless necessary or requested.