# JML_XAI_Project

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://e-strauss.github.io/JML_XAI_Project.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://e-strauss.github.io/JML_XAI_Project.jl/dev/)
[![Build Status](https://github.com/e-strauss/JML_XAI_Project.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/e-strauss/JML_XAI_Project.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/e-strauss/JML_XAI_Project.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/e-strauss/JML_XAI_Project.jl)

# About JML_XAI_Project
The JML_XAI_Project package implements the explainable AI methods SHAP and LIME for image inputs. The project was developed as part of the "Julia for Machine Learning" course at TU Berlin.

[Here](https://e-strauss.github.io/JML_XAI_Project/dev/) you can find the documentation.

# Running Unit Tests
1. **clone the repo and open it in VSCode:**
   - git clone git@github.com:e-strauss/JML_XAI_Project.git
   - open the folder in VSCode or run "code ." in your terminal

2. **Activate Your Package Environment:**
   - Typing Alt+j Alt+o (option+j option+o on macOS) opens a Julia REPL and directly activates your local environment
   - If this doesn't work on the first try, you might have to manually select your "Julia env" [JML_XAI_Project] in the bottom bar of VSCode once
   - Enter package mode by typing `]`

3. **Run the Tests:**
   - While still in package mode, execute the `test` command
