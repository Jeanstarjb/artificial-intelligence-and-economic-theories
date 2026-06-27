# Artificial Intelligence and Economic Theories

This repository provides an implementation and exploration of concepts discussed in the research paper [Artificial Intelligence and Economic Theories](https://arxiv.org/pdf/1703.06597v1) by Tshilidzi Marwala and Evan Hurwitz. The paper examines how advancements in artificial intelligence (AI) can influence and reshape classical economic theories. It bridges the gap between computational techniques inspired by natural intelligence and the foundational principles of economics, offering a fresh perspective on topics such as demand and supply, game theory, market efficiency, and more.

---

## Core Concepts

The paper investigates the intersection of AI and economics by addressing how AI-driven approaches can refine or replace existing economic theories. Below are some of the key economic theories discussed and their connection to AI:

1. **Demand and Supply**: AI models, such as neural networks, can dynamically predict and balance supply-demand relationships in real-time markets.
2. **Asymmetrical Information**: Machine learning algorithms can minimize information asymmetry by extracting hidden patterns from large datasets, thus promoting fairer transactions.
3. **Game Theory**: AI techniques like reinforcement learning optimize decision-making in competitive and cooperative settings.
4. **Efficient Market Hypotheses (EMH)**: The ability of AI to process high-frequency financial data challenges traditional assumptions about market efficiency.
5. **Bounded Rationality**: AI provides a mechanism to model decision-making under incomplete information, aligning with Herbert Simon's bounded rationality framework.
6. **Causality and Counterfactuals**: AI enables a deeper understanding of causal relationships in economic systems, improving policy-making and scenario analysis.

The overarching theme of the paper is to integrate modern AI advancements into traditional economic models, enabling more adaptive, data-driven, and realistic representations of economic behavior.

---

## Repository Overview

This repository contains Python and PyTorch-based implementations and simulations that explore the application of AI techniques to the economic theories discussed in the paper.

### Structure

```
.
├── data/
│   ├── market_data.csv       # Example dataset for supply-demand simulation
│   ├── game_theory_matrix.csv # Payoff matrix for game theory examples
│   └── portfolio_data.csv    # Dataset for portfolio optimization
├── models/
│   ├── demand_supply_model.py # Neural network for supply-demand prediction
│   ├── game_theory_rl.py      # Reinforcement learning for game theory
│   └── portfolio_optimizer.py # Portfolio optimization using AI
├── notebooks/
│   ├── 01_demand_supply.ipynb # Notebook showcasing supply-demand predictions
│   ├── 02_game_theory.ipynb   # Notebook exploring AI in game theory
│   └── 03_portfolio.ipynb     # Notebook on portfolio optimization
├── utils/
│   ├── data_preprocessing.py  # Helper functions for data loading and processing
│   └── visualization.py       # Utility functions for visualizing results
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Features

1. **Demand and Supply Prediction**:
   - A neural network trained on synthetic and real-world market data to predict supply and demand equilibrium points dynamically.
   - Implements supervised learning using PyTorch.

2. **AI in Game Theory**:
   - A reinforcement learning agent that learns optimal strategies in a simulated game.
   - Explores Nash equilibria and other game-theoretic principles using AI.

3. **Portfolio Optimization**:
   - Uses AI techniques to allocate assets in a portfolio based on historical data and user-defined constraints.
   - Implements mean-variance optimization and compares it with AI-enhanced strategies.

4. **Causal Inference**:
   - Demonstrates how causal relationships can be identified and validated using AI, focusing on economic data.

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ai-economic-theories.git
   cd ai-economic-theories
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate   # On Windows: .\env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Jupyter notebooks or code files:
   ```
   jupyter notebook
   ```

---

## Usage

### 1. Demand and Supply Prediction
Navigate to `notebooks/01_demand_supply.ipynb` to train and evaluate a neural network for predicting demand and supply equilibrium. The notebook walks through:
- Loading and preprocessing market data.
- Training a neural network using PyTorch.
- Visualizing the supply-demand equilibrium.

### 2. Game Theory with AI
Explore the application of reinforcement learning in game theory by running `notebooks/02_game_theory.ipynb`. This notebook includes:
- Simulating a game with a predefined payoff matrix.
- Training an agent to learn optimal strategies using Q-learning.
- Analyzing Nash equilibria derived from the AI model.

### 3. Portfolio Optimization
The `notebooks/03_portfolio.ipynb` notebook demonstrates how AI can enhance portfolio optimization:
- Loading historical financial data.
- Applying mean-variance optimization.
- Comparing traditional strategies to AI-enhanced models.

---

## Dependencies

This project requires Python 3.7 or higher. The major dependencies include:

- `torch`: For building and training neural networks.
- `numpy`: For numerical computations.
- `pandas`: For data manipulation and preprocessing.
- `matplotlib` and `seaborn`: For data visualization.
- `scikit-learn`: For additional machine learning utilities.

Refer to `requirements.txt` for the full list of dependencies.

---

## Contributing

Contributions are welcome! If you have suggestions for improving the code or extending the project, feel free to submit a pull request or open an issue.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

This repository is inspired by the ideas presented in the research paper [Artificial Intelligence and Economic Theories](https://arxiv.org/pdf/1703.06597v1) by Tshilidzi Marwala and Evan Hurwitz. Special thanks to the authors for their insightful work on the intersection of AI and economics.