# ProbabilisticMethods
Application for analysis and processing of statistical data by probabilistic methods.
## Building and running the project
- Clone the repository on your PC.
- Run `main.py` file.
## Using the ProbabilisticMethods
### Part 1
The first step is to enter the number of elements to be generated from the given distributions:
1. Probability density:

For $x ≥ 0$:

$$ f(x) = γx^{5s+4}e^{-Ax^5}, γ ∈ R, A ∈ R, s ∈ N $$

For $x < 0$:

$$ f(x) = 0 $$

2. Poisson distribution.

The second step for the generated values ​​is the search for unbiased estimates of the mathematical expectation and variance.

The third step is the search for estimates of distribution parameters by the maximum likelihood method.

The fourth step for the initial distributions is to build intervals $(M(X) - l/9; M(X) + 5l/2)$ the probability of falling into which is equal to $p$.

The fifth step builds the intervals from the fourth step, but for the generated distributions.

### Part 2

