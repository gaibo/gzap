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
