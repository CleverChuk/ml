## Approach:
To perform this task, I did the following:
    Carried out manual model selection by trying out different models on the data set.
    Gradient boosting classifier performed better than all the models investigated including
    Neural network model(MLP).

*Feature Engineering*:
    The feature engineering approaching is very transparent and can be found in transform_data_with_fu
    function. 'money_back_guarantee' was found to reduce performance and was eliminated from
    consideration. The remaining features was then used to train the model.
    Two vectorizers were used: DictVectorizer which does one-hot encoding for categorical features and
    Tfidf vecctorixer with character level analysis for categorical features was utilized. Tfidf performed
    better than dict vectorizer.