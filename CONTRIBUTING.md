# Contributing to Trade-Agent

Thank you for your interest in contributing to Trade-Agent! This document provides guidelines and information for contributors.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:
- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Trade-Agent.git
   cd Trade-Agent
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ViratSrivastava/Trade-Agent.git
   ```

## Development Setup

### Environment Setup
1. Create virtual environment:
   ```bash
   python -m venv stock_rl_env
   source stock_rl_env/bin/activate  # Linux/Mac
   # or
   stock_rl_env\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Configuration
1. Copy environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your API keys and configuration

## Development Workflow

### Branching Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical fixes

### Creating a Feature Branch
```bash
git checkout develop
git pull upstream develop
git checkout -b feature/your-feature-name
```

### Making Changes
1. Write code following the style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass

## Code Style

### Python Style
- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters
- Use Black for code formatting

### Formatting
```bash
# Format code
black src/

# Check style
flake8 src/
```

### Naming Conventions
- Classes: PascalCase
- Functions/methods: snake_case
- Constants: UPPER_CASE
- Private methods: _leading_underscore

### Documentation
- Use docstrings for all public functions/classes
- Follow Google/NumPy docstring format
- Include type hints
- Document complex algorithms

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_trading_environment.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Structure
- Unit tests in `tests/` directory
- Test files named `test_*.py`
- Use pytest fixtures for setup/teardown
- Aim for >80% code coverage

### Writing Tests
```python
import pytest
from src.trading_environment import TradingEnvironment

class TestTradingEnvironment:
    def test_initialization(self):
        env = TradingEnvironment()
        assert env.balance == 10000
        assert env.shares == 0

    def test_buy_action(self):
        env = TradingEnvironment()
        state, reward, done = env.step(1)  # BUY
        assert reward >= 0
```

## Documentation

### Code Documentation
- All public functions need docstrings
- Complex logic should be commented
- Update README for new features

### API Documentation
- Use Sphinx for generating docs
- Keep docstrings up to date
- Document breaking changes

## Submitting Changes

### Pull Request Process
1. Ensure your branch is up to date:
   ```bash
   git fetch upstream
   git rebase upstream/develop
   ```

2. Run tests and linting:
   ```bash
   pytest
   black src/
   flake8 src/
   ```

3. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature

   - Description of changes
   - Related issue: #123"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create Pull Request on GitHub

### PR Guidelines
- Use descriptive titles
- Reference related issues
- Provide clear description of changes
- Include screenshots for UI changes
- Request review from maintainers

### Commit Messages
Follow conventional commits:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

## Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Error messages/logs
- Screenshots if applicable

### Feature Requests
For new features, please include:
- Clear description of the feature
- Use case and benefits
- Implementation suggestions
- Mockups or examples

### Security Issues
- Report security vulnerabilities privately
- Contact maintainers directly
- Do not create public issues for security problems

## Recognition

Contributors will be recognized in:
- CHANGELOG.md
- GitHub contributors list
- Project documentation

Thank you for contributing to Trade-Agent!</content>
<parameter name="filePath">c:\Users\VIRAT\Projects\Trade-Agent\CONTRIBUTING.md