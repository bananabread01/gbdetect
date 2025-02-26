from flask import Flask

app = Flask(__name__)

@app.route('/channels')
def channel():
    return {"channel":["channel1", "channel2", "channel3"]}

if __name__=='__main__':
    app.run(debug=True)