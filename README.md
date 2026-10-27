# Training Strategies and Regularisation

A neural network has many settings besides its weights, from the activation functions to the optimiser and the batch size. In these notebooks you compare these settings by their learning curves, then reduce overfitting with L2 regularisation, dropout and early stopping. A final exercise applies them to real data.

## Learning Objectives

By the end of this repository, you should be able to:

- Choose the output activation and the loss for regression, binary and multiclass problems.
- Compare activations, optimisers, batch sizes and weight initialisers by their validation learning curves, and add batch normalisation to a deep network so that it trains.
- Diagnose underfitting, overfitting and a good fit from training and validation learning curves.
- Apply L2 regularisation, dropout and early stopping to a Keras model, and choose the value early stopping monitors.
- Save the best epoch of a training run with a `ModelCheckpoint` callback and load it again.
- Select a model on the validation set and evaluate it once on the test set.

## Learning Path

Work through the notebooks in this order:

| File / Folder | Description |
|---|---|
| [**01 - Training Strategies for Neural Networks**](01_training_strategies.ipynb) | Compare activations, optimisers, batch sizes, initialisers and batch normalisation. |
| [**02 - Overfitting and Regularisation**](02_overfitting_and_regularisation.ipynb) | Diagnose learning curves and reduce overfitting with L2, dropout, early stopping and checkpoints. |
| [**03 - Exercise: Regularise the Forest Cover Classifier**](03_exercise_forest_cover.ipynb) | Improve a tree species classifier on real data and evaluate it on the test set. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Solutions**](solutions/) | Worked solutions for all three notebooks. |
| [**Assets**](assets/) | The figures used in the notebooks and the scripts that draw them. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
> **placeholder**. Replace it, including the `< >` brackets, with your own
> value. For example, `cd <repo-name>` becomes `cd amle-training-strategies-regularisation`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies, including TensorFlow and Keras, and creates a virtual environment in `.venv/`. TensorFlow is a large package, so the first install takes a few minutes.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**Deep Learning with Python**](https://deeplearningwithpython.io/): François Chollet's book on Keras, free to read online; chapter 5 covers overfitting and regularisation.
- [**Google Machine Learning Crash Course: Interpreting loss curves**](https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves): Practice reading loss curves, with the cause and the fix for each pattern.
- [**An overview of gradient descent optimization algorithms**](https://www.ruder.io/optimizing-gradient-descent/): Sebastian Ruder's explanation of momentum, RMSprop, Adam and their relatives.
- [**Initializing neural networks**](https://www.deeplearning.ai/ai-notes/initialization/index.html): An interactive deeplearning.ai note on why the scale of the starting weights matters.
- [**Dropout: A Simple Way to Prevent Neural Networks from Overfitting**](https://jmlr.org/papers/v15/srivastava14a.html): The original dropout paper by Srivastava, Hinton and colleagues.
- [**TensorFlow tutorial: Overfit and underfit**](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit): Model size, L2 and dropout compared on a larger dataset.
- [**A Recipe for Training Neural Networks**](https://karpathy.github.io/2019/04/25/recipe/): Andrej Karpathy's step-by-step process for training networks in practice, from overfitting a small batch to regularising.
- [**Deep Learning Tuning Playbook**](https://github.com/google-research/tuning_playbook): How researchers at Google choose optimisers, batch sizes and other hyperparameters for real projects.
- [**TensorFlow Playground**](https://playground.tensorflow.org/): Change the activation, learning rate and regularisation of a small network in the browser and watch it train.
