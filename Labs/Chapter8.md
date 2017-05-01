Chapter 8 Lab
================

-   [Setup](#setup)
-   [Decision Trees: Classification](#decision-trees-classification)
    -   [The Choice of Sub-Tree](#the-choice-of-sub-tree)
    -   [Summary](#summary)
-   [Regression Trees](#regression-trees)

Setup
=====

We work on the `Carseats` data, and we are all about growing a tree from carseats!

``` r
rm(list = ls())
search()
```

    ## [1] ".GlobalEnv"        "package:stats"     "package:graphics" 
    ## [4] "package:grDevices" "package:utils"     "package:datasets" 
    ## [7] "package:methods"   "Autoloads"         "package:base"

``` r
library(ISLR)
library(tree)
library(gmodels)
# knowing the data and variables
dim(Carseats)
```

    ## [1] 400  11

``` r
summary(Carseats)
```

    ##      Sales          CompPrice       Income        Advertising    
    ##  Min.   : 0.000   Min.   : 77   Min.   : 21.00   Min.   : 0.000  
    ##  1st Qu.: 5.390   1st Qu.:115   1st Qu.: 42.75   1st Qu.: 0.000  
    ##  Median : 7.490   Median :125   Median : 69.00   Median : 5.000  
    ##  Mean   : 7.496   Mean   :125   Mean   : 68.66   Mean   : 6.635  
    ##  3rd Qu.: 9.320   3rd Qu.:135   3rd Qu.: 91.00   3rd Qu.:12.000  
    ##  Max.   :16.270   Max.   :175   Max.   :120.00   Max.   :29.000  
    ##    Population        Price        ShelveLoc        Age       
    ##  Min.   : 10.0   Min.   : 24.0   Bad   : 96   Min.   :25.00  
    ##  1st Qu.:139.0   1st Qu.:100.0   Good  : 85   1st Qu.:39.75  
    ##  Median :272.0   Median :117.0   Medium:219   Median :54.50  
    ##  Mean   :264.8   Mean   :115.8                Mean   :53.32  
    ##  3rd Qu.:398.5   3rd Qu.:131.0                3rd Qu.:66.00  
    ##  Max.   :509.0   Max.   :191.0                Max.   :80.00  
    ##    Education    Urban       US     
    ##  Min.   :10.0   No :118   No :142  
    ##  1st Qu.:12.0   Yes:282   Yes:258  
    ##  Median :14.0                      
    ##  Mean   :13.9                      
    ##  3rd Qu.:16.0                      
    ##  Max.   :18.0

``` r
sum(is.na(Carseats))  # no missing
```

    ## [1] 0

``` r
# learning about response
summary(Carseats$Sales)  # unit unknown, mean and median 7.5
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   0.000   5.390   7.490   7.496   9.320  16.270

``` r
hist(Carseats$Sales)  # not much skewed
```

![](Chapter8_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
# attaching the data: not recommended, but we follow the ISLR book
attach(Carseats)
```

Decision Trees: Classification
==============================

After the initial setup, we create a dummy variable from `Sales` to work on as the response.

``` r
High = ifelse(Sales > 8, "YES", "No")
summary(High)
```

    ##    Length     Class      Mode 
    ##       400 character character

The variable `High` is a vector of strings. It resides in the Global Environment and that is why `summary(High)` gives us no information about the values it takes. So the next step is to add the variable to the working data:

``` r
Carseats = data.frame(Carseats, High)
```

Time to grow a tree:

``` r
tree.carseats = tree(High ~ . - Sales, data = Carseats)
summary(tree.carseats)
```

    ## 
    ## Classification tree:
    ## tree(formula = High ~ . - Sales, data = Carseats)
    ## Variables actually used in tree construction:
    ## [1] "ShelveLoc"   "Price"       "Income"      "CompPrice"   "Population" 
    ## [6] "Advertising" "Age"         "US"         
    ## Number of terminal nodes:  27 
    ## Residual mean deviance:  0.4575 = 170.7 / 373 
    ## Misclassification error rate: 0.09 = 36 / 400

What is the stopping criteria? The default criterion is used above. The stopping criterion is controlled by the argument `control = tree.control(...)` for `tree()` whose default is to have `tree.control(nobs, mincut = 5, minsize = 10, mindev = 0.01)`.

We see residual mean deviance in the output. How is the residual mean deviance computed? It is equal to two times the ratio of log-likelihoods of the saturated model to the model being considered:

$$
Dev = 2\\times \\frac{\\mathcal{L} ( y|\\theta\_s)}{\\mathcal{L}(y|\\theta\_0)}
= 2\\times \\frac {1} { \\log\\Big(\\prod\_{m,k} \\big({\\hat{p}\_{mk}}^{n\_{mk}}\\big)\\Big) }
= -2\\sum\_m \\sum\_k n\_{mk}\\log \\hat{p}\_{mk}\\,,
$$

where *θ*<sub>*s*</sub> is the vector of parameters for the saturated model, and *θ*<sub>0</sub> is the one for the model we consider (which is of course nested in the saturated model). The probability distributions are assumed to be multimonial. The saturated model is tree with a leaf for every observation. Hence, the saturated tree perfectly fits the data and its estimates $\\hat{p}\_{mk}$ are all equal to 1. The mean deviance is computed by dividing deviance by *n* − |*T*|.
Next, we graphically represent the tree:

``` r
plot(tree.carseats)
text(tree.carseats, pretty = 0, cex = 0.6)
```

![](Chapter8_files/figure-markdown_github/unnamed-chunk-5-1.png) The ability to visualize a tree is one of the most attractive properties of trees. Shelving location appears to be the most important indicator of Sales. The option `pretty = 0` makes the graph display category names, rather than single letters for each category. More details on the tree can be seen by typing the tree's name:

``` r
summary(Carseats$High)
```

    ##  No YES 
    ## 236 164

``` r
tree.carseats
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##   1) root 400 541.500 No ( 0.59000 0.41000 )  
    ##     2) ShelveLoc: Bad,Medium 315 390.600 No ( 0.68889 0.31111 )  
    ##       4) Price < 92.5 46  56.530 YES ( 0.30435 0.69565 )  
    ##         8) Income < 57 10  12.220 No ( 0.70000 0.30000 )  
    ##          16) CompPrice < 110.5 5   0.000 No ( 1.00000 0.00000 ) *
    ##          17) CompPrice > 110.5 5   6.730 YES ( 0.40000 0.60000 ) *
    ##         9) Income > 57 36  35.470 YES ( 0.19444 0.80556 )  
    ##          18) Population < 207.5 16  21.170 YES ( 0.37500 0.62500 ) *
    ##          19) Population > 207.5 20   7.941 YES ( 0.05000 0.95000 ) *
    ##       5) Price > 92.5 269 299.800 No ( 0.75465 0.24535 )  
    ##        10) Advertising < 13.5 224 213.200 No ( 0.81696 0.18304 )  
    ##          20) CompPrice < 124.5 96  44.890 No ( 0.93750 0.06250 )  
    ##            40) Price < 106.5 38  33.150 No ( 0.84211 0.15789 )  
    ##              80) Population < 177 12  16.300 No ( 0.58333 0.41667 )  
    ##               160) Income < 60.5 6   0.000 No ( 1.00000 0.00000 ) *
    ##               161) Income > 60.5 6   5.407 YES ( 0.16667 0.83333 ) *
    ##              81) Population > 177 26   8.477 No ( 0.96154 0.03846 ) *
    ##            41) Price > 106.5 58   0.000 No ( 1.00000 0.00000 ) *
    ##          21) CompPrice > 124.5 128 150.200 No ( 0.72656 0.27344 )  
    ##            42) Price < 122.5 51  70.680 YES ( 0.49020 0.50980 )  
    ##              84) ShelveLoc: Bad 11   6.702 No ( 0.90909 0.09091 ) *
    ##              85) ShelveLoc: Medium 40  52.930 YES ( 0.37500 0.62500 )  
    ##               170) Price < 109.5 16   7.481 YES ( 0.06250 0.93750 ) *
    ##               171) Price > 109.5 24  32.600 No ( 0.58333 0.41667 )  
    ##                 342) Age < 49.5 13  16.050 YES ( 0.30769 0.69231 ) *
    ##                 343) Age > 49.5 11   6.702 No ( 0.90909 0.09091 ) *
    ##            43) Price > 122.5 77  55.540 No ( 0.88312 0.11688 )  
    ##              86) CompPrice < 147.5 58  17.400 No ( 0.96552 0.03448 ) *
    ##              87) CompPrice > 147.5 19  25.010 No ( 0.63158 0.36842 )  
    ##               174) Price < 147 12  16.300 YES ( 0.41667 0.58333 )  
    ##                 348) CompPrice < 152.5 7   5.742 YES ( 0.14286 0.85714 ) *
    ##                 349) CompPrice > 152.5 5   5.004 No ( 0.80000 0.20000 ) *
    ##               175) Price > 147 7   0.000 No ( 1.00000 0.00000 ) *
    ##        11) Advertising > 13.5 45  61.830 YES ( 0.44444 0.55556 )  
    ##          22) Age < 54.5 25  25.020 YES ( 0.20000 0.80000 )  
    ##            44) CompPrice < 130.5 14  18.250 YES ( 0.35714 0.64286 )  
    ##              88) Income < 100 9  12.370 No ( 0.55556 0.44444 ) *
    ##              89) Income > 100 5   0.000 YES ( 0.00000 1.00000 ) *
    ##            45) CompPrice > 130.5 11   0.000 YES ( 0.00000 1.00000 ) *
    ##          23) Age > 54.5 20  22.490 No ( 0.75000 0.25000 )  
    ##            46) CompPrice < 122.5 10   0.000 No ( 1.00000 0.00000 ) *
    ##            47) CompPrice > 122.5 10  13.860 No ( 0.50000 0.50000 )  
    ##              94) Price < 125 5   0.000 YES ( 0.00000 1.00000 ) *
    ##              95) Price > 125 5   0.000 No ( 1.00000 0.00000 ) *
    ##     3) ShelveLoc: Good 85  90.330 YES ( 0.22353 0.77647 )  
    ##       6) Price < 135 68  49.260 YES ( 0.11765 0.88235 )  
    ##        12) US: No 17  22.070 YES ( 0.35294 0.64706 )  
    ##          24) Price < 109 8   0.000 YES ( 0.00000 1.00000 ) *
    ##          25) Price > 109 9  11.460 No ( 0.66667 0.33333 ) *
    ##        13) US: Yes 51  16.880 YES ( 0.03922 0.96078 ) *
    ##       7) Price > 135 17  22.070 No ( 0.64706 0.35294 )  
    ##        14) Income < 46 6   0.000 No ( 1.00000 0.00000 ) *
    ##        15) Income > 46 11  15.160 YES ( 0.45455 0.54545 ) *

Each node is represented with a number. One can trace back a node by dividing this number by two and taking the integer part of it. The resulting number denotes the parent node. By sequentially doing this, we can trace the node back to the root. \* denotes the terminal nodes.
The other information are the number of observations in each branch, the deviance, the overall prediction for the branch, and the fraction of leaves in that branch that take on values "NO" and "YES", respectively.
Why is "NO" presented before "YES"? It is probably due to the way the variable is defined. However, which value is the baseline does not make much of a difference, here (although it makes a difference in other cases, e.g. when we want to interpret linear regression results).

What about the test error rate?

``` r
set.seed(2)
train = sample(nrow(Carseats), 200)
test = -train
test.carseats = Carseats[test, ]
test.high = High[test] 
tree.carseats = tree(High ~ . - Sales, data = Carseats, subset = train)
tree.preds = predict(tree.carseats, newdata = test.carseats, type = "class")
head(tree.preds)
```

    ## [1] No  YES No  No  No  YES
    ## Levels: No YES

``` r
table(tree.preds, test.high)
```

    ##           test.high
    ## tree.preds No YES
    ##        No  86  27
    ##        YES 30  57

``` r
(86+57)/200
```

    ## [1] 0.715

The test error rate estimate is 72%, while the training error rate was estimated to be 91%.

The Choice of Sub-Tree
----------------------

Pruning the tree is done through cross-validation over a sequence of trees found by cost-complexity pruning.

``` r
set.seed(3)
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
```

    ## [1] "size"   "dev"    "k"      "method"

``` r
cv.carseats
```

    ## $size
    ## [1] 19 17 14 13  9  7  3  2  1
    ## 
    ## $dev
    ## [1] 55 55 53 52 50 56 69 65 80
    ## 
    ## $k
    ## [1]       -Inf  0.0000000  0.6666667  1.0000000  1.7500000  2.0000000
    ## [7]  4.2500000  5.0000000 23.0000000
    ## 
    ## $method
    ## [1] "misclass"
    ## 
    ## attr(,"class")
    ## [1] "prune"         "tree.sequence"

The `FUN` argument in `cv.tree` determines the nested sequence of subtrees and is the output of the command `prune.tree()`, which has uses either of the following estimates for error:

1.  Deviance, which is the default
2.  Misclassification error, which is done through equating the argument `FUN` to either `prune.tree(method = "misclass")` or its short form `prune.misclass`

In the output of `cv.tree` above, `$size` is the number of *terminal* nodes, and `$k` corresponds to *α*, the tuning parameter in cost complexity pruning. The output `$dev` corresponds to the error, which is misclassification error in this case, despite its name. The value of *α* = −∞ would be the largest possible tree with RSS = 0, which has 27 leaves here.

``` r
par(mfrow = c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")
```

![](Chapter8_files/figure-markdown_github/unnamed-chunk-9-1.png)

So the tree with 9 terminal nodes results in the lowest cross-validation error rate.

``` r
tree.prune = prune.misclass(tree.carseats, best = 9)
plot(tree.prune)
text(tree.prune, pretty = 0)
```

![](Chapter8_files/figure-markdown_github/unnamed-chunk-10-1.png)

``` r
tree.prune
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##   1) root 200 269.200 No ( 0.6000 0.4000 )  
    ##     2) ShelveLoc: Bad,Medium 153 185.400 No ( 0.7059 0.2941 )  
    ##       4) Price < 142 130 167.700 No ( 0.6538 0.3462 )  
    ##         8) ShelveLoc: Bad 39  29.870 No ( 0.8718 0.1282 ) *
    ##         9) ShelveLoc: Medium 91 124.800 No ( 0.5604 0.4396 )  
    ##          18) Price < 86.5 9   0.000 YES ( 0.0000 1.0000 ) *
    ##          19) Price > 86.5 82 108.700 No ( 0.6220 0.3780 )  
    ##            38) Advertising < 6.5 52  56.180 No ( 0.7692 0.2308 ) *
    ##            39) Advertising > 6.5 30  39.430 YES ( 0.3667 0.6333 )  
    ##              78) Age < 37.5 5   0.000 YES ( 0.0000 1.0000 ) *
    ##              79) Age > 37.5 25  34.300 YES ( 0.4400 0.5600 )  
    ##               158) CompPrice < 118.5 8   8.997 No ( 0.7500 0.2500 ) *
    ##               159) CompPrice > 118.5 17  20.600 YES ( 0.2941 0.7059 ) *
    ##       5) Price > 142 23   0.000 No ( 1.0000 0.0000 ) *
    ##     3) ShelveLoc: Good 47  53.400 YES ( 0.2553 0.7447 )  
    ##       6) Price < 142.5 38  29.590 YES ( 0.1316 0.8684 ) *
    ##       7) Price > 142.5 9   9.535 No ( 0.7778 0.2222 ) *

The pruned tree becomes more interpretable. The test error for the tree with 9 leaves:

``` r
tree.preds = predict(tree.prune, newdata = Carseats[test, ], type = "class")
table(tree.preds, High[test])
```

    ##           
    ## tree.preds No YES
    ##        No  94  24
    ##        YES 22  60

``` r
(94 + 60)/ 200
```

    ## [1] 0.77

The prunin process has not only made the tree more interpretable, but it has also improved the classification accuracy. The prediction accuracy falls if we increase the value of `best`:

``` r
prune.carseats.15 = prune.misclass(tree.carseats, best = 15)
plot(prune.carseats.15)
text(prune.carseats.15, pretty = 0)
```

![](Chapter8_files/figure-markdown_github/unnamed-chunk-12-1.png)

``` r
prune.pred.15 = predict(prune.carseats.15, type = "class", newdata = Carseats[test, ])
tab = table(prune.pred.15, High[test])
sum(diag(tab))/sum(tab)
```

    ## [1] 0.74

Summary
-------

-   Creating qualitative variable: We can use `ifelse` to create new factor variables.
    -   Remember to merge the generated variables with the data set
-   After fitting the tree, we learned how to
    -   plot it
        -   The `plot.tree()` command will plot the tree without text.
        -   How to add details? We use `text.tree()` to add text.
            -   Factors by name: option `pretty = 0`
    -   see its nodes and details: type the name of the tree
    -   compute test error rate
        -   In `predict.tree()`, the argument `type = class` used to get factor predictions
    -   do cost complexity pruning: use `cv.tree` with the argument `FUN = prune.misclass`
        -   `cv.tree(...)$k` denotes the number of terminal nodes
    -   depict the sub-tree with lowest error.
        -   `prune.misclass(tree_name, best = best_leaves)`, where best\_leaves is the number of terminal nodes that result in lowest cross-validation error
-   Remember to set the seed before doing validation set approach or CV

-   Errors:
    -   test error after pruning: computed for the original tree

Regression Trees
================
