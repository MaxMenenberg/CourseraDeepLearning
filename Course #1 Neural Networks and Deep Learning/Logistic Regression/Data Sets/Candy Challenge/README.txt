Content

candy-data.csv includes attributes for each candy along with its ranking. For binary variables, 1 means yes, 0 means no. The data contains the following fields:

    chocolate: Does it contain chocolate?
    fruity: Is it fruit flavored?
    caramel: Is there caramel in the candy?
    peanutalmondy: Does it contain peanuts, peanut butter or almonds?
    nougat: Does it contain nougat?
    crispedricewafer: Does it contain crisped rice, wafers, or a cookie component?
    hard: Is it a hard candy?
    bar: Is it a candy bar?
    pluribus: Is it one of many candies in a bag or box?
    sugarpercent: The percentile of sugar it falls under within the data set.
    pricepercent: The unit price percentile compared to the rest of the set.
    winpercent: The overall win percentage according to 269,000 matchups.


For the purposes of testing logistic regression we will remove try and predict if a certain candy is
chocolate given the other parts of the data