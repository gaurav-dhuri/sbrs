from flask import Flask, render_template, request
import recommend

app = Flask(__name__, template_folder='templates')

NO_USER_ERR = "User Name doesn't exist, No product recommendations currently!"


@app.route('/', methods=['POST', 'GET'])
def home():
    recommended_products = []
    if request.method == 'POST':
        username = request.form['UserName']
        recommended_products = recommend.get_user_recommendations(username)
    if (not (recommended_products is None)):
        return render_template("index.html", product_name_list=recommended_products)
    else:
        return render_template("index.html", message=NO_USER_ERR)


if __name__ == "__main__":
    app.run()
