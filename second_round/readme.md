** Problem 1 **
* I have not calculated evidence probabilities here since they are not really required to get labels,
as we divide all probabilities by evidence probabilities always. Let me know if you wanted those to be present as well.
* Code has some hard coded implementation for this problem which can obviously be further worked upon,
I ran out of time before I could make the api more generic.
* Obviously there are many flavours of NB algorithm and I have implemented only one, this implementation can only deal with categorical data.
* Here is what am trying to do - 
    * Find all unique labels
    * Calculate prior probabilities for all labels
    * Calculate all evidence probabilities (optional)
    * calculate likelihoods of all features for all labels
    * Calculate class probabilities for given points using Baye's theorem using the above values
    * Return the predicted label as the label with highest class probability
* For continuous variables Gaussian distributions seems to be the most common one recommended for Naive bayes, I did some
research and found a Flexible Naive bayes algorithm that uses kernal density estimation instead of Gaussian. Based on my limited
research it also seemed that it is not recommended to use other distributions such as t etc for Naive bayes. Can you explain what
distributions you were thinking of ?


** Problem 2 **
* Implementing KNN was straight forward, but the problem here was the points were sparse.
* I converted them to dense so as to be able to calculate euclidean distances.
* My code is fairly slow as compared to inbuilt implementations of KNN, the reason is obvious,
my code is not optimized to handle sparse matrix so well. I ran a similar code of KNN on another
non sparse data set and found run times were only 1-2 times more expensive then their library implementations.
* I ran the predictions only for first few points because of this, we can run for all, will just take longer.
* Majority voting is been used in KNN to calculate final neighbours among neighbours. In case of tie,
I am simply using the first one. But again we can do more complex stuff here such as apply weights to label, etc
* The output of the custom implementation was exactly same as that for the library implementation.
(Tested on first 10 points [-1. -1.  1.  1. -1. -1. -1.  1. -1. -1.])

** Problem 3 **
* I read the pages of the book that you gave me and a few other articles.
* The approach seems an extension of KNN with the difference that elements instead of contributing equally
towards a prediction, contribute based on their weights which in turn is calculated based on their closeness
to the predictor whose value needs to be predicted.
* Approach -
    * For each xj  in test set -
        * for each xi in train set - 
            * calculate the k((xi -xj)/h)
            * calculate Weights Wi as per the formula
            * Calculate wi*yi
       * yj = sum of (wi * yi)
* For binary classification the sign of yj determines the label (+1, -1)
* For regression yj belongs to R and is an estimate of Yj
            

** General **
* I have not used sklearn as you requested, however I did use pandas a bit for basic IO and array manipulation,
I am guessing that is fine? Numpy sometimes gets tedious with complicated array manipulation.
* I am guessing you are trying to see my understanding of various algorithms. I have also included,
the implementation of Linear regression, Multi dimensional linear regression and Polynomial linear regression.
* Those implementations were written for some custom datasets so might have some hardcoded parameters,
but still should be enough to explain the understanding of the algorithms.
