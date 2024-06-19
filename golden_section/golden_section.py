import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Sequence, Any, TypeVar, Protocol, Optional
mpl.use('Qt5Agg')   # matplotlib 3.5 changed default backend and PyCharm freezes; revert to avoid problem

# Explanation ##################################################################

"""
    Apparently in 1980s China, schools (and agricultural/industrial-focused society) hammered into kids 
the concept of golden-section search, i.e. use of the golden ratio number (1.618 or its inverse 0.618; 
more on that later) to do optimization problems by hand. This came to light when I asked my parents a 
coding interview question, only for them to simultaneously recite "0.618" before I had even laid out details 
(FYI: the answer did not involve golden ratio or even binary search, but rather clever load balancing).
    Baffled by their unfamiliar dependence on the golden ratio (which sounded completely archaic to my 
late-2000s American education), I went on some tirade about how binary search is the end-all in any scenario 
involving search. They suggested I prove it by writing code.
    It was a great idea - I'd knock out a Monte Carlo-type experiment in half an hour and get some coding
practice. In an hour, I had a result saying binary search was about 4% more efficent (fewer average "splits") 
than doing the same with "0.618", and by the end of the night I had added nice documentation and pretty charts. 
I wouldn't realize until a couple months later - when I revisited the code for a write-up - how nonsensically 
contextless the "it" was that I proved.

    This function/script/experiment must now attempt to explain my misunderstanding in the way golden-section 
search works. Because, well first of all, it's not remotely comparable to binary search. Skipping right over
the dozens of hours it took me to piece it together: golden-section search makes perfect sense in the context 
of finding local extrema in functions, a process that requires COMPARING 2 VALUES IN THE SAMPLE to determine
which section to recursively search next. That is virtually unrelated to binary search, where you have A VALUE 
IN HAND to compare to values in a SORTED sample.

    Think of golden-section search as trying to figure out the number of grams of sugar that tastes best in
a vat of iced tea (and, uh, assume your taste buds don't tell you any first derivative info i.e. whether 
you'd like it sweeter at a given level... actually this is a terrible example but whatever, just assume you 
can only compare relative taste levels): you're pretty sure 0g is too bitter, and 1000g is too sweet. You 
might naturally say to sample at 500g - but then all you find out is it tastes way better than 0g and 1000g; 
d'oh, what next? You now naturally say why not compare 333g to 667g - well that works!
[---------------------------------|----------------------------------|---------------------------------]
0                                333                                667                              1000
You may find out both taste better than 0g and 1000g, and 667g tastes better than 333g. So now you can be
pretty certain anything below 333g is not going to beat 333g and definitely won't beat 667g. You can shrink
your range to 0.667 of its original size now.
 ---------------------------------[----------------------|------------|-----------|----------------------]
0                                333                    *555         667         *777                  1000
It's at this point you have a revelation - we already have the 667g taste data point, but it's smack dab
in the middle of our new range, and we don't know which side of 667g will taste better (because of our non-
differentiable taste buds). It seems wasteful to pick 2 new taste points to trisect the range. Wouldn't there 
exist a ratio to split the range where one more taste perfectly maintains the previous split ratio?
                                                   c
                  a                                                    b
[--------------------------------------|------------------------|--------------------------------------]
0                                     382                      618                                   1000
 --------------------------------------[------------------------|--------------|------------------------]
0                                     382                      618            *764                    1000
Yeah so golden ratio is that ratio. The math is pretty straightforward:

a + c = b; a/b = c/(b-c)
let's get rid of c and try to distill a and b into a ratio - it's 3 variables with 2 equations, 
but we can still get a ratio
a/b = (b-a)/(b-(b-a))
a/b = (b-a)/a
a/b = b/a - 1
we have particular interest in relative ratio of a/b, so let's name it n
n + 1 = 1/n     # Remember how 0.618 + 1 = 1/0.618 = 1.618? What a cool property
n^2 + n - 1 = 0
quadratic formula: n = (-1 + sqrt(5))/2 = 0.618
So ratio of short section to long is 0.618, but what if we want to sanity check long to whole?
b/(a+b) = 1/(a/b + 1) = 1/1.618 = 0.618!

That's cool - now you only need 1 taste (see 764g above) and 1 compare per iteration to cut the range
to 0.618 of its previous size. For one more step in the example, say 618g tastes better than 764g:
 --------------------------------------[---------------|---------|--------------]------------------------
0                                     382             *528      618            764                     1000

Appendix? For completion, note that you can try to do a variation with bisection splits:
[--------------------------------------------------|-------------------------|-------------------------]
0                                                 500                       750                      1000
where it winds up either:
 --------------------------------------------------[-------------------------|-------------------------]
0                                                 500                       750                      1000
or:
[--------------------------------------------------|-------------------------]-------------------------
0                                                 500                       750                      1000
In a uniform distribution (certainly not accurate for sugar tasting), these range reductions of 0.5 and 0.75
(750g tasting better or worse) have the same chance, so on average this is a 0.625, which beats trisection
(0.667) but still loses to golden (0.618). Note also something interesting if we still care about number of
tastes - you could naturally alternate between "bisection" and "trisection" (what I'm calling the very first 
example with 333g and 667g) based on whether you wind up left or right. Though these patterns are cool, math 
generally favors the load balancing solution, i.e. the stable golden ratio.

    Anyway, what does relate golden-section search to binary search is a derivative called Fibonacci search. 
This is a confusing hack to do something akin to binary search but without division, because computers sucked 
in the 1950s. It exploits the fact that the ratio of a Fibonacci number to the next approaches the golden ratio, 
and that you can get from one Fib number to another with only addition/subtraction.
Fibonacci numbers: 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987...
[-------------------------------------------------------------|--------------------------------------]
         89  144       233            377                    610                                    987
when it goes left:
[--------------------------------------|-----------------------]--------------------------------------
         89  144       233            *377                    610                                    987
           55     89           144                233                            377
when it goes right:
 -------------------------------------------------------------[-----------------------|---------------]
         89  144       233            377                    610                     *843            987
           55     89           144                233                    233                 144
note that 843 is NOT a Fib number! But it must be this way when we split the right subsection into Fibonacci 
intervals; visually you can see how if we always went left, we'd stay exclusively with actual Fib numbers. 
Fibonacci search is (perhaps obviously) a "worse" version of binary search that was practical on ancient machines 
because it avoided multiplication/division and had possible cache or non-uniform storage access efficiencies. 
Though I jumped straight to the golden ratio instead of coding for the Fibonacci numbers, this type of "variation 
on binary search" is what my first hour result that initial afternoon tried to explain. Emphasis on "tried" - 
it took a whole lot of sitting and thinking afterwards to put it in context. 

    Here lay the next major problem with my quick project - I, uh, maybe didn't fully understand binary search. 
We already had an inkling of that when I revealed it took me hours to understand why golden-section can do 
extrema while binary can't. Specifically, I was stuck visualizing binary search as placing a target number 
into a continuous number line... which is a sub-case, but A TRIVIAL ONE (you know the number!). Technically, 
it tries to find a target value among sorted values by narrowing the range of INDEXES - I didn't even think about 
distribution of array values BEYOND THE INDEXES; I also didn't code for not finding the value (i.e. I used a 
questionable break condition), and I overengineered the logic by trying to have two integer halfway marks.
    Prior to this project, I hadn't mapped in my head that x and f(x) in math can correspond to indexes and array 
values at those indexes, i.e. i and a[i]. This matters because I had written my Monte Carlo experiment to 
just find target values (to my credit, drawn from a distribution)... in a linear index between 0 and 1,000,000. 
Ignoring that that's a trivial task, I hadn't thought about how we can and should map a second distribution to 
that array a[] - search performs very differently when the values mapped to the indexes increase exponentially 
vs. linearly, etc.! 

The new mission statement as I committed to re-writing the code looked something like this:
1) Fix basic binary search algo to use array instead of just domain; simplify algo logic and fix break condition
  - test all the variations of
    "inserting uniformly distributed targets into sorted values with uniformly distributed frequency"
               normally                                              normally
  - consider effect of duplicates and missing values; could minimize "collisions" by making integer range 
    much larger than array size, but then we get a lot of missing target values, which affects our average 
    number of splits metric for efficiency
2) Add functinoality for Fibonacci search - I had all this complicated logic for "inside" and "outside" 
   (see function for concept explanation) but it couldn't replicate the static 1.618:1 left:right setup
"""

# [MANUAL] CONFIGURATIONS ######################################################

# Default example - draw a million integers distributed on [0, 1,000,000]
MIN_INTEGER = 0
MAX_INTEGER = 1_000_000
N_SIMULATIONS = 1_000_000


# Generate random numbers for the simulation ###################################

# Constants
GOLDEN_RATIO = (1 + 5**0.5) / 2     # 1.618...; note that 1/GR == GR-1 == 0.618
# As of numpy 1.17, Generator is preferable function for doing random numbers
rng = np.random.default_rng()   # Random float uniformly distributed over [0, 1)

# Uniform distribution random integers array of size N_SIMULALATIONS
uniform_random = rng.integers(MIN_INTEGER, MAX_INTEGER, endpoint=True, size=N_SIMULATIONS)    # [MIN_INT, MAX_INT]

# Normal distribution random integers array of size N_SIMULALATIONS
mu = (MAX_INTEGER-MIN_INTEGER) / 2    # Set center of bell curve at center of range [MIN_INTEGER, MAX_INTEGER]
sigma = (MAX_INTEGER-MIN_INTEGER) / 6   # Set std such that 3 std each direction (99.7%) is the "bounds" of our range
normal_random = rng.normal(mu, sigma, size=N_SIMULATIONS).clip(MIN_INTEGER, MAX_INTEGER).round()    # Clamp and round

# Visualize integer distributions
fig, ax = plt.subplots()
ax.hist(uniform_random, color='C3', alpha=0.5, label='Uniform Distribution')    # Should look like a rectangle block
ax.hist(normal_random, color='C4', alpha=0.5, label='Normal Distribution (with clamping and rounding)')     # Bell curve
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('Random Number from Generator')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Sanity Check: Here are the numbers we will try to locate via\n"
             f"recursive search algos on range [{MIN_INTEGER:,.0f}, {MAX_INTEGER:,.0f}]")
ax.legend()
fig.tight_layout()
plt.show()  # First figure of script may not show without this


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

# Numpy "vectorized" versions of search algos we can apply to the arrays of numbers
binary_search_apply = np.vectorize(lambda n, a: binary_search(n, a), otypes=[int])
fibonacci_search_apply = np.vectorize(lambda n, a: fibonacci_search(n, a), otypes=[int])
golden_search_inside_apply = np.vectorize(lambda n, a: golden_section_search_inside(n, a), otypes=[int])
golden_search_outside_apply = np.vectorize(lambda n, a: golden_section_search_outside(n, a), otypes=[int])

# Finally, run the search algos on the random numbers
sorted_uniform_random = np.sort(uniform_random)
sorted_normal_random = np.sort(normal_random)
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
                  'Golden (Inside)': [uniform_golden_inside_cuts.mean(), normal_golden_inside_cuts.mean()],
                  'Golden (Outside)': [uniform_golden_outside_cuts.mean(), normal_golden_outside_cuts.mean()]},
                 index=['Uniform', 'Normal'])
print(means_table)

# Visualize number-of-splits distributions - histogram, control for integer bins


def aesthetic_integer_bins(int_arr):
    """ Generate numpy array of bin boundaries given a numpy array of integers to be binned.
        This is surprisingly tricky in default np.histogram() because
        1) final bin is inclusive of final bin boundary rather than exclusive like every other bin right bound
        2) graphically "centering" bin on the integer involves left and right bounds fractionally around integer
    :param int_arr: numpy array of integers to be binned
    :return: numpy array of bin boundaries (usually float, as it includes fractional bounds around integers)
    """
    unique_ints = np.unique(int_arr)
    step = np.min(np.diff(unique_ints))     # Usually 1 when dealing with consecutive integers
    # From half a step below min to half a step above max, allowing each integer to be "centered"
    # Note extra step added to right bound because np.arange() excludes rightmost step
    return np.arange(np.min(unique_ints)-step/2, np.max(unique_ints)+step/2+step, step)


# Uniform
fig, ax = plt.subplots()
ax.hist(uniform_binary_cuts, bins=aesthetic_integer_bins(uniform_binary_cuts), alpha=0.5,
        label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
ax.hist(uniform_golden_inside_cuts, bins=aesthetic_integer_bins(uniform_golden_inside_cuts), alpha=0.5,
        label=f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Uniform', 'Golden (Inside)')]:.2f}")
ax.hist(uniform_golden_outside_cuts, bins=aesthetic_integer_bins(uniform_golden_outside_cuts), alpha=0.5,
        label=f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Uniform', 'Golden (Outside)')]:.2f}")
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
