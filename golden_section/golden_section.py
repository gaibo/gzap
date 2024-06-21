import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Sequence, Any, TypeVar, Protocol, Optional
mpl.use('Qt5Agg')   # matplotlib 3.5 changed default backend and PyCharm freezes; revert to avoid problem


# Explanation ##################################################################

"""
This section has been moved to the independent README. 

One thing I'll re-emphasize here is that while I created this file with the 
name golden_section.py and the intention to explore golden-section search's 
relation to binary search, I actually found out that's all wrong! 

I implemented a "generalized" binary search and tested binary search against 
its Golden ratio-using variant, Fibonacci search. The results are all about that. 

I did not in the end implement golden-section search proper, because it turns out 
that was never the original disagreement! Let this be a record of how arguments 
can arise from "sophisticated misunderstanding"!
"""


# [MANUAL] CONFIGURATIONS ######################################################

"""
Default experiment:
Sample a million integers distributed over [0, 100,000,000] as "targets" to find
via search in a sorted array of size 10,000,000 that also comes from a distribution 
over the same interval.
The "simulation" is thus 1,000,000 searches over various distribution scenarios.

- Note that we control the likelihood a given number may not even be in the array. 
  In this example, the 100,000,001 possible integers is 10x the size of the array, 
  so for example with uniform distributions there is a 1/10 chance of actually 
  finding a given target in the array. The rest of the searches will go "all the 
  way" to the worst case scenario number of splits - in this sense, this would 
  control how much we weight and punish worst case performance for a given search.
  This is one mechanic through which the default numbers would affect analysis 
  of efficiency of the different searches!

- Note that this means we must choose one distribution from which to sample "targets" 
  and another distribution from which to sample values for "sorted arrays"!
  - For my setup, each can be "uniform" or "normal" - that's 4 combinations.

- I specifically set up the default numbers to avoid confusing conflations. However, 
  note that if we had an additional constraint that the array size is the same as 
  the number of integers in the interval (e.g. array of size 1,000,001, interval 
  [0, 1,000,000]), we could construct a "trivial" case where we let the sorted 
  array's values be equal to their indexes (e.g. 0 to 1,000,000). This is trivial 
  in the sense that indexing a target number into the array actually wouldn't even 
  require search.
  - This is simply a special sub-case of filling the sorted array with the uniform 
    distribution, so I will not specifically build for it, as intuition can be 
    gained from existing results.
"""
# TODO: How would results be affected from tweaking these numbers?
MIN_INTEGER, MAX_INTEGER = 0, 100_000_000
ARRAY_SIZE = 10_000_000
N_SIMULATIONS = 1_000_000   # i.e. Number of targets


# Generate random numbers for the simulation ###################################

# Constants
GOLDEN_RATIO = (1 + 5**0.5) / 2     # 1.618...; note that 1/GR == GR-1 == 0.618
# As of numpy 1.17, Generator is preferable function for doing random numbers
RNG = np.random.default_rng()   # Random float uniformly distributed over [0, 1)

# Uniform distribution random integers for "targets" and sorted array
TARGETS_UNIFORM = RNG.integers(MIN_INTEGER, MAX_INTEGER, endpoint=True, size=N_SIMULATIONS)
# NOTE: I will just reuse the same sorted array for all N_SIMULATIONS... that's fine right?
ARRAY_UNIFORM = np.sort(RNG.integers(MIN_INTEGER, MAX_INTEGER, endpoint=True, size=ARRAY_SIZE))

# Normal distribution random integers for "targets" and sorted array
mu = (MAX_INTEGER-MIN_INTEGER) / 2    # Set center of bell curve at center of range [MIN_INTEGER, MAX_INTEGER]
sigma = (MAX_INTEGER-MIN_INTEGER) / 6   # Set std such that 3 std each direction (99.7%) is the "bounds" of our range
TARGETS_NORMAL = RNG.normal(mu, sigma, size=N_SIMULATIONS).clip(MIN_INTEGER, MAX_INTEGER).round()   # Clamp and round
ARRAY_NORMAL = np.sort(RNG.normal(mu, sigma, size=ARRAY_SIZE).clip(MIN_INTEGER, MAX_INTEGER).round())

# Visualize target integer distributions
fig, ax = plt.subplots()
ax.hist(TARGETS_UNIFORM, color='C3', alpha=0.5, label='Uniform Distribution')    # Should look like a rectangle block
ax.hist(TARGETS_NORMAL, color='C4', alpha=0.5, label='Normal Distribution (with clamping and rounding)')    # Bell curve
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('Random Number from Generator')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Sanity Check: Here are the \"target\" integers we will try to locate\n"
             f"via recursive search algos on range [{MIN_INTEGER:,.0f}, {MAX_INTEGER:,.0f}]")
ax.legend()
fig.tight_layout()
plt.show()  # First figure of script may not show without this

# Visualize array integer distributions
fig, ax = plt.subplots()
ax.hist(ARRAY_UNIFORM, color='C3', alpha=0.5, label='Uniform Distribution')    # Should look like a rectangle block
ax.hist(ARRAY_NORMAL, color='C4', alpha=0.5, label='Normal Distribution (with clamping and rounding)')    # Bell curve
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('Random Number from Generator')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Sanity Check: Here are the integers filling the \"sorted arrays\"\n"
             f"inside which we will search for the \"targets\"")
ax.legend()
fig.tight_layout()


# Functions ####################################################################

# NOTE: This is me trying to practice Python type-hinting and generic types...
class SupportsLT(Protocol):
    def __lt__(self, __other: Any, /) -> bool: ...  # Double underscore for positional-only arg


CT = TypeVar('CT', bound=SupportsLT)    # "Comparable" generic type! (in Python 3 implementations all you need is <)


def section_search(target: CT, sorted_arr: Sequence[CT], split: float = 0.5, use_insideoutside: bool = False,
                   split_insideoutside: str = 'inside', verbose: bool = True) -> Optional[int]:
    """ Generalization of binary search such that the split (0.5 for binary) may be changed.
        This becomes somewhat convoluted as it introduces asymmetry - do you want the 0.7 split on left or right?
        (Fibonacci search for example, for Fibonacci sequence math reasons, keeps ~0.618 split on left side always.)
        My answer is to introduce concept of always keeping split fraction "inside" or "outside", which requires
        storing state on whether the previous split sent you "left" or "right". e.g.,

        [----------] (10 difference between bounds), split=0.7; initial inside/outside N/A, previous left/right N/A
        Split 1 - arbitrary because no state; let it be "inside", as if previous "right": [-------|---]
        Split 2a - inside, previous left:  [--|-----]
        Split 2b - inside, previous right:          [--|-]
        Split 2c - outside, previous left: [-----|--]
        Split 2d - outside, previous right:         [-|--]

        I'm imagining "inside" vs. "outside" as a way someone would try to bias splits
        to benefit in certain distributions - if you know numbers near the center of the
        bounds are more common, could you get there faster by playing with this parameter?
        Answer: Empirically, a split >0.5 does better on normal distribution with "outside" vs. "inside", but both
                are significantly worse than just binary search. e.g., on a million random numbers [0, 1,000,000],
                Mean # of Splits  Binary 0.5  Golden 0.618 (Inside)  Golden 0.618 (Outside)
                Uniform            19.951814              20.701467               20.701008
                Normal             19.948935              20.987887               20.528782
                Note inside/outside perform the exact same on a uniform distribution, as you would expect!
                On uniform, Fibonacci search (0.618) is about 3.8% worse here than binary search (0.5).
    :param target: (comparable) value to search for (within the array)
    :param sorted_arr: pre-sorted array of (indexable, comparable) values
    :param split: fractional split of bounds; set 0.5 for binary search
    :param use_insideoutside: set True to use my persistent "inside/outside" split system
    :param split_insideoutside: "inside/outside" split concept defined in description above
    :param verbose: set True for print statements describing splits
    :return: integer number of comparisons (think splits) needed to arrive at target within sorted array
             or None if target is not found within array
    """
    assert 0 < split < 1
    assert split_insideoutside in ['inside', 'outside']

    # Initialize
    i_left: int = 0  # Left lower bound index
    i_right: int = len(sorted_arr) - 1   # Right upper bound index
    completed_comparisons = 0
    prev_leftright = 'right'    # Arbitrarily intialized; note difference it makes when switching inside/outside

    # Perform binary search until failure condition: boundary indexes overlap (still need to check at index if equals)
    while i_left <= i_right:
        split_size_floor = int(split * (i_right - i_left))  # "floor" because left/right confusing later

        # Complicated cases describing how someone might use a ratio split in relation to "center" of range
        if (use_insideoutside is False
                or (split_insideoutside == 'inside' and prev_leftright == 'right')
                or (split_insideoutside == 'outside' and prev_leftright == 'left')):
            # Put split from left to right - e.g. split=0.7 -> [-------|---]
            i_mid: int = i_left + split_size_floor     # int in [i_left, i_right-1]
        elif ((split_insideoutside == 'inside' and prev_leftright == 'left')
              or (split_insideoutside == 'outside' and prev_leftright == 'right')):
            # Put split from right to left - e.g. split=0.7 -> [---|-------]
            i_mid: int = i_right - split_size_floor    # int in [i_left+1, i_right]
        else:
            raise ValueError(f"IMPOSSIBLE: ratio split case ('{split_insideoutside}', '{prev_leftright}')")

        # Split towards target's appropriate bounds
        value_left: CT = sorted_arr[i_left]
        value_right: CT = sorted_arr[i_right]
        value_mid: CT = sorted_arr[i_mid]
        completed_comparisons += 1  # Not done yet, but want this ahead of verbose
        if verbose:
            print(f"Evaluation {completed_comparisons}: target {target}")
            print(f"\tindexes: [{i_left}, {i_mid}] | [{i_mid}, {i_right}]")
            print(f"\tvalues:  [{value_left}, {value_mid}] | [{value_mid}, {value_right}]")
        if target < value_mid:
            if verbose:
                print(f"Went LEFT!")
            i_right = i_mid - 1     # May now violate i_right > i_left!
            prev_leftright = 'left'
        elif target > value_mid:
            if verbose:
                print(f"Went RIGHT!")
            i_left = i_mid + 1  # May now violate i_left < i_right!
            prev_leftright = 'right'
        else:
            if verbose:
                print(f"****FOUND {value_mid} at index {i_mid} after {completed_comparisons} comparisons!")
            return completed_comparisons

    # Unsuccessful at finding target in sorted_arr
    if verbose:
        print(f"****UNFOUND {target} after all {completed_comparisons} comparisons!")
    return None


def binary_search(target, sorted_arr, verbose=False):
    return section_search(target, sorted_arr, 0.5, use_insideoutside=False, verbose=verbose)


def fibonacci_search(target, sorted_arr, verbose=False):
    # Not strictly Fibonacci search because I skip straight to golden ratio...
    # Note that default use_insideoutside=False puts split on the left, as desired
    return section_search(target, sorted_arr, split=1/GOLDEN_RATIO, use_insideoutside=False, verbose=verbose)


def golden_section_search_inside(target, sorted_arr, verbose=False):
    return section_search(target, sorted_arr, split=1/GOLDEN_RATIO, use_insideoutside=True,
                          split_insideoutside='inside', verbose=verbose)


def golden_section_search_outside(target, sorted_arr, verbose=False):
    return section_search(target, sorted_arr, split=1/GOLDEN_RATIO, use_insideoutside=True,
                          split_insideoutside='outside', verbose=verbose)


# Run the simulations ##########################################################

# Experiment variables
search_types = ['binary', 'fibonacci', 'golden_inside', 'golden_outside']
search_dispatch = {
    'binary': binary_search,
    'fibonacci': fibonacci_search,
    'golden_inside': golden_section_search_inside,
    'golden_outside': golden_section_search_outside
}
array_types = ['uniform', 'normal']
array_dispatch = {
    'uniform': ARRAY_UNIFORM,
    'normal': ARRAY_NORMAL
}
target_types = ['uniform', 'normal']
target_dispatch = {
    'uniform': TARGETS_UNIFORM,
    'normal': TARGETS_NORMAL
}

# Numpy "vectorized" versions of search algos we can apply to the arrays of targets
apply_search_to_targets_dict = {}
for search_type in search_types:
    for array_type in array_types:
        apply_search_to_targets_dict[search_type, array_type] = \
            np.vectorize(lambda n: search_dispatch[search_type](n, array_dispatch[array_type]), otypes=[float])

# Finally, run the search algos on the targets and record results
results_dict = {}
for target_type in target_types:
    for search_type in search_types:
        for array_type in array_types:
            print(f"Calculating {target_type}, {search_type}, {array_type}...")
            results_dict[target_type, search_type, array_type] = \
                apply_search_to_targets_dict[search_type, array_type](target_dispatch[target_type])
            print("Done.")


# Results - Binary vs. Golden ##################################################

# Table
means_table = \
    pd.DataFrame({'Binary': [np.nanmean(results_dict['uniform', 'binary', 'uniform']),
                             np.nanmean(results_dict['uniform', 'binary', 'normal']),
                             np.nanmean(results_dict['normal', 'binary', 'uniform']),
                             np.nanmean(results_dict['normal', 'binary', 'normal'])],
                  'Fibonacci': [np.nanmean(results_dict['uniform', 'fibonacci', 'uniform']),
                                np.nanmean(results_dict['uniform', 'fibonacci', 'normal']),
                                np.nanmean(results_dict['normal', 'fibonacci', 'uniform']),
                                np.nanmean(results_dict['normal', 'fibonacci', 'normal'])],
                  'Golden (Inside)': [np.nanmean(results_dict['uniform', 'golden_inside', 'uniform']),
                                      np.nanmean(results_dict['uniform', 'golden_inside', 'normal']),
                                      np.nanmean(results_dict['normal', 'golden_inside', 'uniform']),
                                      np.nanmean(results_dict['normal', 'golden_inside', 'normal'])],
                  'Golden (Outside)': [np.nanmean(results_dict['uniform', 'golden_outside', 'uniform']),
                                       np.nanmean(results_dict['uniform', 'golden_outside', 'normal']),
                                       np.nanmean(results_dict['normal', 'golden_outside', 'uniform']),
                                       np.nanmean(results_dict['normal', 'golden_outside', 'normal'])]},
                 index=['Uniform Targets, Uniform Array',
                        'Uniform Targets, Normal Array',
                        'Normal Targets, Uniform Array',
                        'Normal Targets, Normal Array'])
print(means_table)

# NOTE: This is a sanity check on how many targets actually exist in the sorted array; constant across each row...
n_table = \
    pd.DataFrame({'Binary': [pd.Series(results_dict['uniform', 'binary', 'uniform']).dropna().shape[0],
                             pd.Series(results_dict['uniform', 'binary', 'normal']).dropna().shape[0],
                             pd.Series(results_dict['normal', 'binary', 'uniform']).dropna().shape[0],
                             pd.Series(results_dict['normal', 'binary', 'normal']).dropna().shape[0]],
                  'Fibonacci': [pd.Series(results_dict['uniform', 'fibonacci', 'uniform']).dropna().shape[0],
                                pd.Series(results_dict['uniform', 'fibonacci', 'normal']).dropna().shape[0],
                                pd.Series(results_dict['normal', 'fibonacci', 'uniform']).dropna().shape[0],
                                pd.Series(results_dict['normal', 'fibonacci', 'normal']).dropna().shape[0]],
                  'Golden (Inside)': [pd.Series(results_dict['uniform', 'golden_inside', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['uniform', 'golden_inside', 'normal']).dropna().shape[0],
                                      pd.Series(results_dict['normal', 'golden_inside', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['normal', 'golden_inside', 'normal']).dropna().shape[0]],
                 'Golden (Outside)': [pd.Series(results_dict['uniform', 'golden_outside', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['uniform', 'golden_outside', 'normal']).dropna().shape[0],
                                      pd.Series(results_dict['normal', 'golden_outside', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['normal', 'golden_outside', 'normal']).dropna().shape[0]]},
                 index=['Uniform Targets, Uniform Array',
                        'Uniform Targets, Normal Array',
                        'Normal Targets, Uniform Array',
                        'Normal Targets, Normal Array'])
print(n_table)

# Visualize number-of-splits distributions - histogram, control for integer bins


def aesthetic_integer_bins(int_arr):
    """ Generate numpy array of bin boundaries given a numpy array of integers to be binned.
        This is surprisingly tricky in default np.histogram() because
        1) final bin is inclusive of final bin boundary rather than exclusive like every other bin right bound
        2) graphically "centering" bin on the integer involves left and right bounds fractionally around integer
    :param int_arr: numpy array of integers to be binned
    :return: numpy array of bin boundaries (usually float, as it includes fractional bounds around integers)
    """
    if int_arr.dtype != int:
        # Try removing NaNs - easiest way is to use pandas dropna()
        int_arr = pd.Series(int_arr).dropna().astype(int)   # Explicitly cast to int as well
    unique_ints = np.unique(int_arr)
    step = np.min(np.diff(unique_ints))     # Usually 1 when dealing with consecutive integers
    # From half a step below min to half a step above max, allowing each integer to be "centered"
    # Note extra step added to right bound because np.arange() excludes rightmost step
    return np.arange(np.min(unique_ints)-step/2, np.max(unique_ints)+step/2+step, step)


# Uniform
# TODO: Figure out which charts/stats are actually important!
fig, ax = plt.subplots()
# ax.hist(uniform_binary_cuts, bins=aesthetic_integer_bins(uniform_binary_cuts), alpha=0.5,
#         label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
# ax.hist(uniform_golden_inside_cuts, bins=aesthetic_integer_bins(uniform_golden_inside_cuts), alpha=0.5,
#         label=f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Uniform', 'Golden (Inside)')]:.2f}")
# ax.hist(uniform_golden_outside_cuts, bins=aesthetic_integer_bins(uniform_golden_outside_cuts), alpha=0.5,
#         label=f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Uniform', 'Golden (Outside)')]:.2f}")
ax.hist(results_dict['uniform', 'binary', 'uniform'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'uniform']), alpha=0.5,
        label=f"Binary (0.5) Uniform-Uniform; Mean {means_table.loc[('Uniform Targets, Uniform Array', 'Binary')]:.2f}")
ax.hist(results_dict['uniform', 'binary', 'normal'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'normal']), alpha=0.5,
        label=f"Binary (0.5) Uniform-Normal; Mean {means_table.loc[('Uniform Targets, Normal Array', 'Binary')]:.2f}")
ax.hist(results_dict['normal', 'binary', 'uniform'], bins=aesthetic_integer_bins(results_dict['normal', 'binary', 'uniform']), alpha=0.5,
        label=f"Binary (0.5) Normal-Uniform; Mean {means_table.loc[('Normal Targets, Uniform Array', 'Binary')]:.2f}")
ax.hist(results_dict['normal', 'binary', 'normal'], bins=aesthetic_integer_bins(results_dict['normal', 'binary', 'normal']), alpha=0.5,
        label=f"Binary (0.5) Normal-Normal; Mean {means_table.loc[('Normal Targets, Normal Array', 'Binary')]:.2f}")

# ax.hist(results_dict['uniform', 'binary', 'uniform'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'uniform']), alpha=0.5,
#         label=f"Binary (0.5) Uniform-Uniform; Mean {means_table.loc[('Uniform Targets, Uniform Array', 'Binary')]:.2f}")
# ax.hist(results_dict['uniform', 'binary', 'normal'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'normal']), alpha=0.5,
#         label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
# ax.hist(results_dict['uniform', 'binary', 'uniform'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'uniform']), alpha=0.5,
#         label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
# ax.hist(results_dict['uniform', 'binary', 'normal'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'normal']), alpha=0.5,
#         label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
# ax.hist(uniform_golden_inside_cuts, bins=aesthetic_integer_bins(uniform_golden_inside_cuts), alpha=0.5,
#         label=f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Uniform', 'Golden (Inside)')]:.2f}")
# ax.hist(uniform_golden_outside_cuts, bins=aesthetic_integer_bins(uniform_golden_outside_cuts), alpha=0.5,
#         label=f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Uniform', 'Golden (Outside)')]:.2f}")
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('# of Splits to Arrive at Number')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Results: Uniform Distribution, {N_SIMULATIONS:,.0f} Samples\n"
             f"Ratio splits of [0, {MAX_INTEGER:,.0f}] needed to arrive at random number")
ax.legend()
fig.tight_layout()

# Normal
fig, ax = plt.subplots()
ax.hist(normal_binary_cuts, bins=aesthetic_integer_bins(normal_binary_cuts), alpha=0.5,
        label=f"Binary (0.5); Mean {means_table.loc[('Normal', 'Binary')]:.2f}")
ax.hist(normal_golden_inside_cuts, bins=aesthetic_integer_bins(normal_golden_inside_cuts), alpha=0.5,
        label=f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Normal', 'Golden (Inside)')]:.2f}")
ax.hist(normal_golden_outside_cuts, bins=aesthetic_integer_bins(normal_golden_outside_cuts), alpha=0.5,
        label=f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Normal', 'Golden (Outside)')]:.2f}")
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('# of Splits to Arrive at Number')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Results: Normal Distribution, {N_SIMULATIONS:,.0f} Samples\n"
             f"Ratio splits of [0, {MAX_INTEGER:,.0f}] needed to arrive at random number")
ax.legend()
fig.tight_layout()

# Print statement
binary_supremacy_uniform = \
    (means_table.loc[('Uniform', 'Golden (Outside)')] - means_table.loc[('Uniform', 'Binary')]) \
    / means_table.loc[('Uniform', 'Binary')]
binary_supremacy_normal = \
    (means_table.loc[('Normal', 'Golden (Outside)')] - means_table.loc[('Normal', 'Binary')]) \
    / means_table.loc[('Normal', 'Binary')]
print(f"Under uniform distribution,\n"
      f"  binary search is {binary_supremacy_uniform*100:.2f}% more efficient\n"
      f"  than Fibonacci search over [0, {MAX_INTEGER:,.0f}], {N_SIMULATIONS:,.0f} simulations.")
print(f"Under normal distribution,\n"
      f"  binary search is {binary_supremacy_normal*100:.2f}% more efficient\n"
      f"  than Fibonacci search over [0, {MAX_INTEGER:,.0f}], {N_SIMULATIONS:,.0f} simulations.")
