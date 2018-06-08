Kaggle TalkingData AdTracking Fraud Detection Challenge 121st's solution (silver medal, top 4%)
===

About CV:
---
We choosed Day6 16:00:00 to Day7 15:59:59 as the 1st day;<br>
We choosed Day7 16:00:00 to Day8 15:59:59 as the 2nd day;<br>
We choosed Day8 16:00:00 to Day9 15:59:59 as the 3rd day;<br>
We added 8 hours to time, thus transformed a day from 16:00:00-15:59:59 into 0:00:00-23:59:59. As a result, we now have 3 complete days. Let's call them tr_7, tr_8, tr_9.<br>
Then we used tr_8 for training, and tr_9 for validation, since we had limited RAM, and we would like to use some features like conversion rate.<br>
After we trained a model with tr_8 for training, and tr_9 for validation, we finished tuning, and we got the best iteration round. Then, we used tr_8+tr_9 for training without any validation. We used the same parameters, and set the num_round to 1.1× the best iteration round we mentioned before. We would have a new model(train on tr_8+tr_9), and we used this model to predict the test data.<br>

About Feature Engineering:
---
"Next click" was absolutely the magic feature! Besides, we used normal aggregate functions such as count or nunique.<br>
In addtion, we used conversion rate, we calculated the rate on tr_7, and merged it on tr_8; calculated on tr_8, and merged it on tr_9;  calculated on tr_9, and merged it on test.<br>
Moreover, numpy can be very useful and way faster than pandas, especially in groupby functions.

About Model:
---
We only used an LGB single model. We tried rank_avg with other models, but it didn't work... We could not figure out the reason.

