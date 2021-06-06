from flask import Flask, render_template, request
import dbn as d
import  feature_extraction as fe

app = Flask(__name__)


@app.route('/', methods =["GET", "POST"])
def get_bot_response():
    if request.method == "POST":
        data = request.form.get('msg')
        features = []

        features.append(fe.featureExtraction(data))

        finalOutput_DBN, reconstructedOutput_DBN = d.dbn.dbn_output(features)

        yhat = d.clf.predict(finalOutput_DBN)

        if(yhat == 1):
           return render_template("phishing.html")
        else:
            return render_template("legitimate.html")
    return render_template("index.html")


if __name__=='__main__':
    app.run()
