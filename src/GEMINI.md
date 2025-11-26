# GEMINI Instructions for src/

This file provides context and instructions for working within the `src/` directory of the Trade-Agent project.

## Subfolder Purpose
- **data_processing/**: Data loading, cleaning, feature engineering, and technical indicators.
- **models/**: Model definitions, training, validation, and related utilities.
- **environment/**: Trading environment and simulation code.
- **visualization/**: Plotting, reporting, and visualization scripts.
- **tests/**: Unit and integration tests for the codebase.

## Import Conventions
- Use absolute imports reflecting the subfolder structure (e.g., `from models.dqn_model import DQNAgent`).
- Avoid relative imports for clarity and maintainability.

## Adding New Code
- Place new scripts in the most relevant subfolder.
- Update this file if you add new subfolders or major features.

## Best Practices
- Keep code modular and well-documented.
- Write tests for new features and bug fixes.
- Use logging for important events and errors.
