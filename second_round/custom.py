# Calculate a linear regression fit
#  train  = 2*1 dataframe containing training data
# Returns value of beta0 and beta1
def linear_reg_fit(train):
    X = train[:, 0]
    Y = train[:, 1]
    beta1 =  np.sum((X-np.mean(X)) * (Y-np.mean(Y)))/np.sum( (X-np.mean(X))**2)
    beta0 = np.mean(Y) - beta1*np.mean(X)
    return beta0, beta1


# Given a linear regression model perform a fit and predict values
# test = test data (n*1)
# beta0, beta1 = Regression coefficients
# Returns predicted y values
def linear_reg_predict(test, beta0, beta1):
    Y = beta0 + beta1 * test
    return np.vstack((test, Y)).T

### Functions for fitting and evaluating multiple linear regression

#--------  multiple_linear_regression_fit
# A function for fitting a multiple linear regression
# Fitted model: f(x) = x.w + c
# Input:
#      x_train (n x d array of predictors in training data)
#      y_train (n x 1 array of response variable vals in training data)
# Return:
#      w (d x 1 array of coefficients)
#      c (float representing intercept)

def multiple_linear_regression_fit(x_train, y_train):

    # Append a column of one's to x
    n = x_train.shape[0]
    ones_col = np.ones((n, 1))
    x_train = np.concatenate((x_train, ones_col), axis=1)

    # Compute transpose of x
    x_transpose = np.transpose(x_train)

    # Compute coefficients: w = inv(x^T * x) x^T * y
    # Compute intermediate term: inv(x^T * x)
    # Note: We have to take pseudo-inverse (pinv), just in case x^T * x is not invertible
    x_t_x_inv = np.linalg.pinv(np.dot(x_transpose, x_train))

    # Compute w: inter_term * x^T * y
    w = np.dot(np.dot(x_t_x_inv, x_transpose), y_train)

    # Obtain intercept: 'c' (last index)
    c = w[-1]

    return w[:-1], c

#--------  multiple_linear_regression_score
# A function for evaluating R^2 score and MSE
# of the linear regression model on a data set
# Input:
#      w (d x 1 array of coefficients)
#      c (float representing intercept)
#      x_test (n x d array of predictors in testing data)
#      y_test (n x 1 array of response variable vals in testing data)
# Return:
#      r_squared (float)
#      y_pred (n x 1 array of predicted y-vals)

def multiple_linear_regression_score(w, c, x_test, y_test):
    # Compute predicted labels
    y_pred = np.dot(x_test, w) + c

    # Evaluate sqaured error, against target labels
    # sq_error = \sum_i (y[i] - y_pred[i])^2
    sq_error = np.sum(np.square(y_test - y_pred))

    # Evaluate squared error for a predicting the mean value, against target labels
    # variance = \sum_i (y[i] - y_mean)^2
    y_mean = np.mean(y_test)
    y_variance = np.sum(np.square(y_test - y_mean))

    # Evaluate R^2 score value
    r_squared = 1 - sq_error / y_variance

    return r_squared, y_pred
### Functions for fitting and evaluating polynomial linear regression
#--------  polynomial_regression_fit
# A function for fitting a polynomial regression
#
# Input:
#      x_train (n x d array of predictors in training data)
#      y_train (n x 1 array of response variable vals in training data)
# Return:
#      w (d x 1 array of coefficients)
#      c (float representing intercept)
def polynomial_regression_fit(x_train, y_train, degrees):
    # Create the poly terms for x,x^2 ..

    n = np.size(y_train)   # data size
    x_poly = np.zeros([n, degrees]) # poly degree

    for d in range(1, degrees +1):
        x_poly[:, d - 1] = np.power(x_train, d)  # adding terms

    Xt = sm.add_constant(x_poly)
    model = sm.OLS(y_train, Xt)
    model_results = model.fit()
    w = model_results.params
    c = w[-1]
    return w[:-1], c

### Functions for predicting polynomial regression
#--------  polynomial_regression_predict
# A function for predicting a polynomial regression
#
# Input:
#      x_test (n x d array of predictors in training data)
#      params (n x 1 array of containing model params coef + incercept)
#      degrees (integer representing the degree of fit)
# Return:
#      y_pred (n x 1 array of predictions)
def polynomial_regression_predict(params, degrees, x_test):
    # # Create the poly terms for x,x^2 ..
    n = x_test.shape[0]
    x_poly = np.zeros([n, degrees])
    for d in range(1, degrees + 1):
        x_poly[:, d - 1] = np.power(x_test, d)
    Xt = sm.add_constant(x_poly)

    # Predict y-vals
    y_pred = np.dot(params, Xt.T)

    return y_pred.T

### Functions for calculating polynomial regression score
#--------  polynomial_regression_score
# A function for calculating polynomial regression score
#
# Input:
#      Y_P (n x 1 array of predictions)
#      Y (n x 1 array of actual response values)
#
# Return:
#      score (R2 score)
def polynomial_regression_score(Y_P, Y):
    RSS = np.sum((Y_P - Y) ** 2)
    TSS = np.sum((Y - np.mean(Y)) ** 2)
    score = 1 - (RSS/TSS)
    return score