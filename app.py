from flask import Flask, render_template, request, url_for, jsonify

app = Flask(__name__)

@app.route('/index.html')
def hello_world3():
    return render_template("index.html")

if __name__ =="__main__":
    app.run(debug=True)