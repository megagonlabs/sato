import os

import pandas as pd
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from predict import evaluate

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# a simple page that says hello
@app.route('/hello')
def hello():
    return 'Hello, World!'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('process_file',
                                    filename=filename))
    return 
    '''
    <!doctype html>
    <title>Upload new file. </title>
    <h1>Upload new File. This is a simple demo, please don't upload any sensitive data.</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/')
def upload():
    return render_template('upload.html')


def extract_feature(df):
    pass

@app.route('/predict',  methods=['POST'])
def predict():


    filename = request.form.get('fId')
    
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    res = evaluate(df)

    return render_template('table.html',  tables=[df.to_html(classes='data', max_rows=30, index=False, header=True)],
                            prediction=res,
                            showButton=False)


@app.route('/uploads/<filename>')
def process_file(filename):
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    
    return render_template('table.html', 
                            tables=[df.to_html(classes='data', max_rows=30, index=False, header=True)],
                            filename=filename,
                            showButton=True)




if __name__ == '__main__':
   app.run()
