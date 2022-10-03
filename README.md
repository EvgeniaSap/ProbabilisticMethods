# ProbabilisticMethods
Application for analysis and processing of statistical data by probabilistic methods.
## Building and running the project
- Clone the repository on your PC (or download and extract the archive).
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

$$ P(X=k)= {1 \over 5} {λ_{1}^k \over k!} e^{-λ_{1}} + {4 \over 5} {λ_{2}^k \over k!}  e^{-λ_{2}} $$

The fourth step is to find the confidence interval for the mathematical expectation. We assume that there is a normal distribution of a random variable, provided that the confidence reliability $𝛾$ is a parameter. Next, the unbiased estimate of the variance is calculated.

The fifth step is to find the confidence interval for the mathematical expectation, provided that the variance is unknown. It is assumed that the original data is normally distributed. Confidence reliability $𝛾$ is a parameter.

The sixth step is to test the hypothesis about the normal distribution of the data.

The seventh step implements an algorithm that allows predicting the future value with a certain probability. The algorithm is developed on the basis of Chebyshev's theorem:

$$ P(|x-a|≥l) ≤ {σ^2 \over l^2}  $$

## Additional Information
- `Student.txt` - Student distribution table.
- `hi.txt` - 
- `laplas.txt` - table of values of the Laplace function.

