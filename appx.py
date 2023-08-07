from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np             
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import  accuracy_score
from tensorflow.keras import datasets

app = Flask(__name__)

dic = {0 : 'airplane', 1 : 'automobile', 2 :'bird', 3:'cat', 4 :'deer', 5 :'dog', 6 :'frog', 7 :'horse', 8 :'ship', 9 :'truck'}



(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_test=x_test/255.0
x_train=x_train/255.0


model = load_model('final2_model.h5')             
model.make_predict_function()                     
 # Making a prediction
y_pred = model.predict(x_test)

 #Taking maximum predicted outcome vlaue
y_classes = [np.argmax(element) for element in y_pred]
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(32,32))           
	i = image.img_to_array(i)
	i = i.reshape(1, 32,32,3)                     

	predict=model.predict(i) 
	p=np.argmax(predict,axis=-1)

	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def Akash():
	return render_template("upload.html")            

@app.route("/about")
def about_page():
	return "About You..!!!"





@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
          Accurancy_score= accuracy_score(y_test, y_classes)
          Precision_Score =precision_score(y_test,y_classes,pos_label='positive', average='micro')
          Recall_Score =recall_score(y_test,y_classes,pos_label='positive',average='macro')
          F1_score= f1_score(y_test,y_classes,pos_label='positive',average='macro')
          img = request.files['my_image']
          img_path="static/" + img.filename
          img.save(img_path)
          p=predict_label(img_path)
          
          
          return render_template("upload.html", a=Accurancy_score,pr=Precision_Score,r=Recall_Score,f=F1_score,prediction = p, img_path = img_path)            
          
          
    

if __name__ =='__main__':
	
	app.run(debug = True)
    

