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
Files `EURCB_161209_211210.txt` and `EURCB_211113_211210.txt` store daily data on the Ruble/Euro exchange rate for the last 5 years (2016-2021) from open sources.

The first step is to set the number of intervals for building an interval variation series.

The second step is the construction of the statistical distribution function.

The third step is the search for point estimates of the distribution parameters by the method of moments. It is assumed that the sample was obtained from a random variable 𝑋, which has a double Poisson distribution:

$$ P(X=k)=1/5  (λ_1^k)/k! e^(-λ_1 )+4/5  (λ_2^k)/k! e^(-λ_2 ) $$

