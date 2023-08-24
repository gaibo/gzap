import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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

    This function/script/experiment is my attempt to explain my misunderstanding of the way golden-section 
search works. Because, well first of all, it's not remotely comparable to binary search. Skipping right over
the dozens of hours it took me to piece it together: golden-section search makes perfect sense in the context 
of finding local extrema in functions, a process that requires COMPARING 2 VALUES IN THE SAMPLE to determine
which section to recursively search next. That is virtually unrelated to binary search, where you have A VALUE 
IN HAND to compare to values in a SORTED sample.

    Think of golden-section search as trying to figure out the number of grams of sugar that tastes best in
a vat of iced tea (and, uh, assume your taste buds don't tell you any first derivative info like whether 
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
It is (perhaps obviously) a "worse" version of binary search that was practical on ancient machines because
it avoided multiplication/division and had possible cache or non-uniform storage access efficiencies. Though 
I jumped straight to the golden ratio instead of coding for the Fibonacci numbers, this type of "variation on
binary search" is what my first hour result that initial afternoon tried to explain. Emphasis on "tried".

    Here lay the next major problem with my quick project - I, uh, maybe didn't fully understand binary search. 
We already had an inkling of that when I revealed it took me hours to understand why golden-section can do 
extrema while binary can't. Specifically, I was stuck thinking that binary search just places a target number
within a domain... which is kind of true, but NOT ENTIRELY, BECAUSE THAT'S TRIVIAL. Technically, it tries 
to find a target value among sorted values by narrowing the range of indexes - I didn't even think about 
distribution of array values BEYOND THE INDEXES; I also didn't code for not finding the value (i.e. I used a 
questionable break condition), and I overengineered the logic by trying to have two pivots.
    Prior to this, I hadn't mapped out in my head that x and f(x) in math can correspond to indexes and array 
values at those indexes, i.e. i and a[i]. This matters because I had written my Monte Carlo experiment to 
just find target values (drawn from a distribution)... in a linear index between 0 and 1,000,000. Ignoring that 
that's a trivial task, I hadn't thought about how we can and should map a second distribution to that array a[] 
- search performs very differently when the values mapped to the indexes increase exponentially vs. linearly! 

The new mission statement as I committed to re-writing the code looked something like this:
1) Fix basic binary search algo to use array instead of just domain; simplify logic and fix break condition
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

def _section_search(number, left_bound, right_bound, split=0.5, use_insideoutside=False, split_insideoutside='inside',
                    prev_leftright='right', completed_sections=0, verbose=False):
    """ Generalization of integer binary search such that the split (0.5 for binary) may be changed.
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
                On uniform, golden-section search (0.618) is about 3.8% worse here than binary search (0.5).
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

    # Recursion break condition - in theory, you should never need to check whether left and right bounds equal
    # if you check will always reach
    if number == left_bound or number == right_bound:
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
                  'Golden (Inside)': [uniform_golden_inside_cuts.mean(), normal_golden_inside_cuts.mean()],
                  'Golden (Outside)': [uniform_golden_outside_cuts.mean(), normal_golden_outside_cuts.mean()]},
                 index=['Uniform', 'Normal'])
print(means_table)

# Visualize # splits distributions - histogram, control for integer bins


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

# It is at this point that I went and actually looked up Golden section search/Fibonacci search.
# I believe the actual Fibonacci search "intelligently" chooses between "inside" and "outside"
# at each step, so it is somehow even more complicated than my version at the moment.
# I will need to check this and fully understand how it works!
# TODO: I think Fibonacci is actually dumber than that - it's just my thing but without consistent inside/outside
