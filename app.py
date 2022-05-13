from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route("/")
def home():
	return render_template("home.html")

@app.route("/check")
def check():
	fs = float(request.args.get("fs"))
	r1 = request.args.get("r1")
	if r1 == "no":
		fu = 0
	else :
		fu = 1
	data = [[fs,fu]]
	
	with open("db.model", "rb") as f :
		model = pickle.load(f)

	res = model.predict(data)
	return render_template("home.html", msg=res)

if __name__ == "__main__":
	app.run(debug=True, use_reloader= True)
	
	
	