from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifierWithCoef(RandomForestClassifier):
    """Adds feature weights for each returned variable"""
    def fit(self, *args, **kwargs):
        """Overloaded fit method to include the feature importances
        of each variable. This is used for RFECV
        """
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
