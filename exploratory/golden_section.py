import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')   # matplotlib 3.5 changed default backend and PyCharm freezes; revert to avoid problem

# [MANUAL] CONFIGURATIONS ######################################################

MAX_INTEGER = 1_000_000     # Implicit MIN_INTEGER = 0
N_SIMULATIONS = 1_000_000


# Generate random numbers for the simulation ###################################

# Constants
GOLDEN_RATIO = (1 + 5**0.5) / 2     # 1.618...; note that 1/GR == GR-1
# As of numpy 1.17, Generator is preferable function for doing random numbers
rng = np.random.default_rng()   # Random float uniformly distributed over [0, 1)

# Uniform distribution random integers array of size N_SIMULALATIONS
uniform_random = rng.integers(0, MAX_INTEGER, endpoint=True, size=N_SIMULATIONS)    # [0, MAX_INTEGER]

# Normal distribution random integers array of size N_SIMULALATIONS
mu = MAX_INTEGER / 2    # Set center of bell curve at center of range [0, MAX_INTEGER]
sigma = MAX_INTEGER / 6   # Set std such that 3 std each direction (99.7%) is the "bounds" of our range
normal_random = rng.normal(mu, sigma, size=N_SIMULATIONS).clip(0, MAX_INTEGER).round()  # Clamp and round to integer

# Visualize integer distributions
fig, ax = plt.subplots()
ax.hist(uniform_random, alpha=0.5, label='Uniform Distribution')    # Should look like a rectangle block
ax.hist(normal_random, alpha=0.5, label='Normal Distribution (with clamping and rounding)')     # Bell curve
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('Random Number from Generator')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Sanity Check: Here are the numbers we will try to locate via\n"
             f"recursive search algos on range [0, {MAX_INTEGER:,.0f}]")
ax.legend()
fig.tight_layout()


# Functions ####################################################################

def _section_search(number, left_bound, right_bound, split=0.5, split_insideoutside='inside',
                    prev_leftright='right', completed_sections=0, verbose=False):
    """ Generalization of integer binary search such that the split (0.5 for binary) may be changed.
        This becomes somewhat convoluted as it introduces asymmetry - do you want the 0.7 split on left or right?
        My answer is to introduce concept of "inside" and "outside" while storing state on whether
        the previous split sent you "left" or "right". e.g.,

        [----------] (10 difference between bounds), split=0.7; initial inside/outside N/A, previous left/right N/A
        Split 1 - arbitrary because no state; let it be "inside", as if previous "right": [-------|---]
        Split 2a - inside, previous left:  [---|----]
        Split 2b - inside, previous right:          [--|-]
        Split 2c - outside, previous left: [----|---]
        Split 2d - outside, previous right:         [-|--]

        I'm imagining "inside" vs. "outside" as a way someone would try to bias splits
        to benefit in certain distributions - if you know numbers near the center of the
        bounds are more common, could you get there faster by playing with this parameter?
        Answer: Empirically, a split >0.5 does better on normal distribution with "outside" vs. "inside", but both
                are significantly worse than just binary search. e.g., on a million random numbers [0, 1,000,000],
                Mean # of Splits  Binary 0.5  Golden 0.618 (Inside)  Golden 0.618 (Outside)
                Uniform            19.951814              20.701467               20.701008
                Normal             19.948935              20.987887               20.528782
                Note that they perform the exact same on a uniform distribution!
    :param number: the integer to search for (within the bounds)
    :param left_bound: smaller integer boundary
    :param right_bound: larger integer boundary
    :param split: fractional split of bounds; set 0.5 for binary search
    :param split_insideoutside: "inside/outside" split concept defined in description above
    :param prev_leftright: state used during recursion - "previous left/right" concept defined in description above
    :param completed_sections: state used during recursion - how many splits have already been done on initial range
    :param verbose: set True for print statements describing splits
    :return: integer number of splits needed to arrive at number within [left_bound, right_bound]
    """
    assert left_bound <= number <= right_bound
    assert 0 < split < 1
    assert split_insideoutside in ['inside', 'outside']
    assert prev_leftright in ['left', 'right']

    # Recursion break condition - will always reach because 2 integer pivots decreases distance between bounds
    if left_bound == right_bound:
        return completed_sections

    # Complicated cases describing how someone might use a ratio split in relation to "center" of range
    split_size_floor = int(split * (right_bound - left_bound))  # Account for possibility of non-int via 2 pivots later
    if ((split_insideoutside == 'inside' and prev_leftright == 'right')
            or (split_insideoutside == 'outside' and prev_leftright == 'left')):
        # Put split from left to right - e.g. split=0.7 -> [-------|---]
        pivot_left = left_bound + split_size_floor
        pivot_right = pivot_left + 1    # Next int
    elif ((split_insideoutside == 'inside' and prev_leftright == 'left')
            or (split_insideoutside == 'outside' and prev_leftright == 'right')):
        # Put split from right to left - e.g. split=0.7 -> [---|-------]
        pivot_right = right_bound - split_size_floor
        pivot_left = pivot_right - 1    # Previous int
    else:
        raise ValueError(f"IMPOSSIBLE: ratio split case ('{split_insideoutside}', '{prev_leftright}')")

    # Split towards number's appropriate bounds
    if verbose:
        print(f"Evaluation {completed_sections+1}:")
        print(f"[{left_bound}, {pivot_left}] | [{pivot_right}, {right_bound}]")
    if number <= pivot_left:
        if verbose:
            print(f"Went LEFT!")
        return _section_search(number, left_bound, pivot_left, split, split_insideoutside,
                               'left', completed_sections+1, verbose)
    elif number >= pivot_right:
        if verbose:
            print(f"Went RIGHT!")
        return _section_search(number, pivot_right, right_bound, split, split_insideoutside,
                               'right', completed_sections+1, verbose)
    else:
        raise ValueError(f"IMPOSSIBLE: number ({number}) could not split using "
                         f"the 2 integer pivots: ...{pivot_left}][{pivot_right}...")


def binary_search(number, left_bound, right_bound, verbose=False):
    return _section_search(number, left_bound, right_bound, 0.5, 'inside', 'right', 0, verbose)


def golden_section_search_inside(number, left_bound, right_bound, verbose=False):
    return _section_search(number, left_bound, right_bound, 1/GOLDEN_RATIO, 'inside', 'right', 0, verbose)


def golden_section_search_outside(number, left_bound, right_bound, verbose=False):
    # Note I choose "outside", "left" so (only) first split matches that of golden_section_search_inside()
    return _section_search(number, left_bound, right_bound, 1/GOLDEN_RATIO, 'outside', 'left', 0, verbose)


# Run the simulations ##########################################################

# Numpy "vectorized" versions of search algos we can apply to the arrays of numbers
binary_search_apply = np.vectorize(lambda n: binary_search(n, 0, MAX_INTEGER), otypes=[int])
golden_search_inside_apply = np.vectorize(lambda n: golden_section_search_inside(n, 0, MAX_INTEGER), otypes=[int])
golden_search_outside_apply = np.vectorize(lambda n: golden_section_search_outside(n, 0, MAX_INTEGER), otypes=[int])

# Finally, run the search algos on the random numbers
uniform_binary_cuts = binary_search_apply(uniform_random)
normal_binary_cuts = binary_search_apply(normal_random)
uniform_golden_inside_cuts = golden_search_inside_apply(uniform_random)
normal_golden_inside_cuts = golden_search_inside_apply(normal_random)
uniform_golden_outside_cuts = golden_search_outside_apply(uniform_random)   # Identical to "inside" Golden for uniform
normal_golden_outside_cuts = golden_search_outside_apply(normal_random)


# Results - Binary vs. Golden ##################################################

# Table
means_table = \
    pd.DataFrame({'Binary': [uniform_binary_cuts.mean(), normal_binary_cuts.mean()],
                  'Golden ("Inside")': [uniform_golden_inside_cuts.mean(), normal_golden_inside_cuts.mean()],
                  'Golden ("Outside")': [uniform_golden_outside_cuts.mean(), normal_golden_outside_cuts.mean()]},
                 index=['Uniform', 'Normal'])

# Visualize # splits distributions - histogram
# Uniform
fig, ax = plt.subplots()
ax.hist(uniform_binary_cuts, alpha=0.5, label='Binary 0.5')
ax.hist(uniform_golden_inside_cuts, alpha=0.5, label='Golden 0.618 ("Inside")')
ax.hist(uniform_golden_outside_cuts, alpha=0.5, label='Golden 0.618 ("Outside")')
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('# of Splits to Arrive at Number')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Results: Uniform Distribution, {N_SIMULATIONS} Samples\n"
             f"Ratio splits of [0, {MAX_INTEGER:,.0f}] needed to arrive at random number")
ax.legend()
fig.tight_layout()
# Normal
fig, ax = plt.subplots()
ax.hist(normal_binary_cuts, alpha=0.5, label='Binary 0.5')
ax.hist(normal_golden_inside_cuts, alpha=0.5, label='Golden 0.618 ("Inside")')
ax.hist(normal_golden_outside_cuts, alpha=0.5, label='Golden 0.618 ("Outside")')
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('# of Splits to Arrive at Number')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Results: Normal Distribution, {N_SIMULATIONS} Samples\n"
             f"Ratio splits of [0, {MAX_INTEGER:,.0f}] needed to arrive at random number")
ax.legend()
fig.tight_layout()

# Print statement
binary_supremacy_uniform = \
    (means_table.loc[('Uniform', 'Golden ("Outside")')] - means_table.loc[('Uniform', 'Binary')]) \
    / means_table.loc[('Uniform', 'Binary')]
binary_supremacy_normal = \
    (means_table.loc[('Normal', 'Golden ("Outside")')] - means_table.loc[('Normal', 'Binary')]) \
    / means_table.loc[('Normal', 'Binary')]
print(f"Under uniform distribution,\n"
      f"  binary search is {binary_supremacy_uniform:.2f}% more efficient\n"
      f"  than Fibonacci search over [0, {MAX_INTEGER}], {N_SIMULATIONS} simulations.")
print(f"Under normal distribution,\n"
      f"  binary search is {binary_supremacy_normal:.2f}% more efficient\n"
      f"  than Fibonacci search over [0, {MAX_INTEGER}], {N_SIMULATIONS} simulations.")

# It is at this point that I went and actually looked up Golden section search/Fibonacci search.
# I believe the actual Fibonacci search "intelligently" chooses between "inside" and "outside"
# at each step, so it is somehow even more complicated than my version at the moment.
# I will need to check this and fully understand how it works!
