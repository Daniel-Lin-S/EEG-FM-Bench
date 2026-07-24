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

## Visualisations

- Use `seaborn` whenever possible rather than `matplotib`
- Scatter plots: set `set_xlim` and (or) `set_ylim` to focus on the main data range.
- Font size must be large enough to be visible. This includes title, axes labels and ticks, figure annotations, legend labels, etc.
- Use lines thicker than default, but not too thick that distracts the main data.
- Use `bbox_to_anchor` and `loc'`  to ensure legends are not overlapping with the main figure, not going beyond figure. Legends should have no frame: `frameon=False`.
- Figures should be coloured, but figure legends and labels should make sense even in black-and-white prints. (for example, use different line types or hatches)
- Titles must be wrapped by `textwrap` with suitable width to prevent the title from exceeding boundaries, and title should be kept away from the main figure with at least one line's gap.
- Print a message after saving a figure, pointing to its file path. If a group of images are plotted, please ONLY print ONE message pointing to their common folder. 
- Proper units should be added to the x and y axes if applicable. (DONT add units if no unit)
- For bar plot (or plot styles where too many samples could create visual clutter): the figure size and font size of axis ticks should be adjusted based on the number of items.