# GEMINI Project Instructions

This file provides specific instructions and context for the Trade-Agent project.

## Project Structure
- **src/**: Main source code, organized into subfolders by functionality (data_processing, models, environment, visualization, tests).
- **data/**: Contains daily and intraday stock data CSVs.
- **models/**: Stores model checkpoints and final weights.
- **notebooks/**: Jupyter notebooks for data analysis and model exploration.
- **docker/**: Docker-related files for containerization.
- **logs/**: Training and validation logs.

## Key Guidelines
- Use the subfolder structure in `src/` for all new scripts.
- Update import statements to reflect the new structure (e.g., `from models.training import ModelTrainer`).
- Place all tests in `src/tests/`.
- Add new data processing scripts to `src/data_processing/`.
- Add new model code to `src/models/`.
- Use `src/environment/` for environment and simulation code.
- Use `src/visualization/` for plotting and visualization utilities.

## Contribution
- Follow the guidelines in `CONTRIBUTING.md`.
- Document new modules and functions clearly.
- Ensure all code passes tests before committing.

## Contact
For questions, refer to `README.md` or contact the maintainers listed there.
