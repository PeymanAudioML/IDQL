# VSCode Configuration for IDQL_pytorch

This directory contains VSCode configuration files for developing and debugging the IDQL_pytorch project.

## Files Overview

### `settings.json`
Configures the Python environment and editor settings:
- Sets the Python interpreter to the virtual environment (`venv/bin/python`)
- Enables linting with flake8
- Configures black formatter
- Sets up pytest for testing
- Excludes virtual environment and cache files from search

### `launch.json`
Debugging configurations for various training scripts:
- **Python: Current File** - Debug the currently open Python file
- **Train DDPM-IQL Offline** - Debug the DDPM-IQL offline training script
- **Train DDPM-IQL Finetune** - Debug the DDPM-IQL finetuning script
- **Train IQL** - Debug the IQL training script
- **Train Offline (States)** - Debug state-based offline training
- **Train Diffusion Offline (States)** - Debug diffusion-based offline training
- **Python: Debug Tests** - Debug pytest tests
- **Python: Attach to Process** - Attach debugger to a running Python process

### `tasks.json`
Build and run tasks:
- **Install Dependencies** - Install packages from requirements.txt
- **Install Package (Editable)** - Install the package in editable mode
- **Run Current Python File** - Execute the currently open Python file
- **Train DDPM-IQL Offline** - Run DDPM-IQL offline training
- **Train DDPM-IQL Finetune** - Run DDPM-IQL finetuning
- **Train IQL (One Param)** - Run IQL training
- **Train Offline (States)** - Run state-based offline training
- **Train Diffusion Offline (States)** - Run diffusion-based offline training
- **Run Tests** - Execute pytest tests
- **Clean Python Cache** - Remove all __pycache__ directories
- **Setup Environment (Full)** - Install dependencies and package (default build task)

## Quick Start

### 1. Activate Virtual Environment
The virtual environment is automatically configured in VSCode. To manually activate it in a terminal:

```bash
source venv/bin/activate
```

### 2. Running Tasks
- Press `Ctrl+Shift+B` (or `Cmd+Shift+B` on macOS) to run the default build task
- Press `Ctrl+Shift+P` and type "Tasks: Run Task" to see all available tasks

### 3. Debugging
- Open a Python file you want to debug
- Press `F5` to start debugging with the default configuration
- Or press `Ctrl+Shift+D` to open the Debug view and select a specific configuration

### 4. Running Tests
- Press `Ctrl+Shift+P` and type "Python: Discover Tests"
- Click the test icon in the left sidebar to view and run tests

## Environment Details

**Python Interpreter:** `${workspaceFolder}/venv/bin/python`

**Installed Packages:**
- PyTorch (CPU version)
- NumPy, SciPy
- Gym, DM Control, MuJoCo
- Weights & Biases (wandb)
- TensorBoard
- MoviePy, ImageIO
- And other dependencies from requirements.txt

## Troubleshooting

### Virtual Environment Not Recognized
If VSCode doesn't recognize the virtual environment:
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `./venv/bin/python`

### Dependencies Not Found
Run the "Setup Environment (Full)" task or manually:
```bash
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e .
```

### Import Errors
Ensure PYTHONPATH is set correctly. It should be automatically configured, but you can verify in the terminal:
```bash
echo $PYTHONPATH
```

## Additional Configuration

### Adding Custom Tasks
Edit `tasks.json` to add new tasks for your specific needs.

### Adding Debug Configurations
Edit `launch.json` to add new debugging configurations.

### Changing Editor Settings
Edit `settings.json` to customize the editor behavior.