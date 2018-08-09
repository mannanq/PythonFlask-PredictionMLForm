from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sklearn.cross_validation
#import height

app = Flask(__name__, template_folder = 'templates') #place holder for current module

@app.route('/')
def heightOut():
    df = pd.read_csv('height.csv')
    x = df['Weight'].values
    y = df['Height'].values
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    lm = LinearRegression()
    lm.fit(x,y)
    #print('Int', lm.intercept_)
    #print('Coef:', lm.coef_)
    plt.scatter(y, lm.predict(x))
    plt.xlabel("Actual Height")
    plt.ylabel("Predicted Height")
    plt.title("Actual vs Predicted Height")
    #plt.show()
    lm.predict(x)
    X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(x, y, test_size = 0.33, random_state = 5)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(Y_train.shape)
    #print(Y_test.shape)

    lm2 = LinearRegression()
    lm2.fit(X_train, Y_train)
    pred_train = lm2.predict(X_train)
    pred_test = lm2.predict(X_test)
    #print("R squared:", lm2.score(X_test, Y_test))

    #plt.scatter(lm2.predict(X_train), lm2.predict(X_train) - Y_train, c = 'b', alpha = 0.5)
    #plt.scatter(lm2.predict(X_test), lm2.predict(X_test) - Y_test, c = 'g')
    #plt.hlines(y = 0, xmin = 50, xmax = 80)
    #plt.title("Residual Plot using training and test data")
    #plt.ylabel("Residuals")
    #plt.show()

    #print(type(lm.intercept_))
    #print(type(lm.coef_))
    intercept = lm.intercept_[0]
    coef = lm.coef_[[0]]
    intercept.astype(int)
    coef.astype(int)
    #print(type(intercept))
    #print(type(coef))
    UserWe = input('Enter Weight:')
    def PredictHeight(UserWe):
        Height = (float(UserWe) *coef) + intercept
        #print('Predicted height of the person is:', Height)
        return Height

    print(PredictHeight(UserWe))
    # if request.method == 'POST' :
    #     result = request.form
    #     return render_template("template.html")

    return render_template("template.html", output=PredictHeight(UserWe))
    # return render_template("with_button.html")


# @app.route('/')
# def formdata():
#     return render_template("with_button.html")

if __name__ == '__main__':
     app.run(debug = True)
