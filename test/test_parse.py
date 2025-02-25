# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2025/2/13 11:12
# @File: test_parse
# @Email: mlshenkai@163.com
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


text = r"""
    To determine the coefficient of \(x^2y^6\) in the expansion of \(\left(\frac{3}{5}x - \frac{y}{2}\right)^8\), we can use the binomial theorem. The binomial theorem states: \[ (a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k \] In this case, \(a = \frac{3}{5}x\), \(b = -\frac{y}{2}\), and \(n = 8\). We are interested in the term that contains \(x^2y^6\). In the general term of the binomial expansion: \[ \binom{8}{k} \left(\frac{3}{5}x\right)^{8-k} \left(-\frac{y}{2}\right)^k \] To get \(x^2\), we need \(8 - k = 2\), thus \(k = 6\). Substituting \(k = 6\) into the expression: \[ \binom{8}{6} \left(\frac{3}{5}x\right)^{8-6} \left(-\frac{y}{2}\right)^6 = \binom{8}{6} \left(\frac{3}{5}x\right)^2 \left(-\frac{y}{2}\right)^6 \] Now, we will compute each part of this expression. 1. Calculate the binomial coefficient \(\binom{8}{6}\). 2. Compute \(\left(\frac{3}{5}\right)^2\). 3. Compute \(\left(-\frac{y}{2}\right)^6\). 4. Combine everything together to get the coefficient of \(x^2y^6\). Let's compute these in Python. ```python from math import comb # Given values n = 8 k = 6 # Calculate the binomial coefficient binom_coeff = comb(n, k) # Compute (3/5)^2 a_term = (3/5)**2 # Compute (-1/2)^6 b_term = (-1/2)**6 # Combine terms to get the coefficient of x^2y^6 coefficient = binom_coeff * a_term * b_term print(coefficient) ``` ```output 0.1575 ``` The coefficient of \(x^2y^6\) in the expansion of \(\left(\frac{3}{5}x - \frac{y}{2}\right)^8\) is \(0.1575\). To express this as a common fraction, we recognize that: \[ 0.1575 = \frac{1575}{10000} = \frac{63}{400} \] Thus, the coefficient can be expressed as: \[ \boxed{\frac{63}{400}} \]
"""
text1 = [r"\frac{63}{400}"]
completion = [[{"content": r"\boxed{\frac{63}{400}}"}]]
solution = [r"\boxed{\frac{63}{400}}"]

rewards = accuracy_reward(completion, [text])
print(rewards)




