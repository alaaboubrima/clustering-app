from PyQt5.QtWidgets import *
import sys,pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.io import arff
import linear_reg,svm_model,table_display,data_visualise,SVR,logistic_reg,RandomForest
import KNN,mlp,pre_trained,add_steps,gaussian


class error_window(QMainWindow):
    def __init__(self):
        super(error_window, self).__init__()
        #uic.loadUi("../ui_files/error.ui", self)
        #self.show()



class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi("ui_files\Mainwindow.ui", self)
 
        # find the widgets in the xml file
 
        #self.textedit = self.findChild(QTextEdit, "textEdit")
        #self.button = self.findChild(QPushButton, "pushButton")
        #self.button.clicked.connect(self.clickedBtn)
        global data,steps
        data=data_visualise.data_()
        steps=add_steps.add_steps()


        
        self.Elbow_btn = self.findChild(QPushButton,"Elbow")
        self.Elbow2_btn = self.findChild(QPushButton,"Elbow2")

        
        
        self.Browse = self.findChild(QPushButton,"Browse")
        self.Drop_btn = self.findChild(QPushButton,"Drop")

        self.fillna_btn = self.findChild(QPushButton,"fill_na")
        self.con_btn = self.findChild(QPushButton,"convert_btn")
        self.columns= self.findChild(QListWidget,"column_list")
        self.emptycolumn=self.findChild(QComboBox,"empty_column")
        self.cat_column=self.findChild(QComboBox,"cat_column")
        self.table = self.findChild(QTableView,"tableView")
        self.dropcolumns=self.findChild(QComboBox,"dropcolumn")
        self.data_shape = self.findChild(QLabel,"shape")
        self.fillmean_btn = self.findChild(QPushButton,"fillmean")
        self.submit_btn = self.findChild(QPushButton,"Submit")
        self.target_col =self.findChild(QLabel,"target_col")
        self.model_select=self.findChild(QComboBox,"model_select")
        #self.describe=self.findChild(QPlainTextEdit,"describe")
        #self.describe= self.findChild(QTextEdit,"Describe")
        
        self.scatter_x=self.findChild(QComboBox,"scatter_x")
        self.scatter_y=self.findChild(QComboBox,"scatter_y")
        self.scatter_mark=self.findChild(QComboBox,"scatter_mark")
        self.scatter_c=self.findChild(QLineEdit,"scatter_c")
        self.hist_k=self.findChild(QLineEdit,"hist_k")

        self.kmean_btn = self.findChild(QPushButton,"kmeanplot")
        
        self.plot_x=self.findChild(QComboBox,"plot_x")
        self.plot_y=self.findChild(QComboBox,"plot_y")
        self.plot_mark=self.findChild(QComboBox,"plot_marker")
        self.plot_c=self.findChild(QLineEdit,"plot_c")
        self.kmedoid_btn = self.findChild(QPushButton,"kmedoidplot")

        self.agnes_k=self.findChild(QLineEdit,"agnes_k")
        self.diana_k=self.findChild(QLineEdit,"diana_k")

        self.agnes_btn = self.findChild(QPushButton,"agnesplot")
        self.diana_btn = self.findChild(QPushButton,"dianaplot")
        self.dbscan_btn = self.findChild(QPushButton,"dbscanplot")
        self.perf_btn = self.findChild(QPushButton,"performance")



        self.min_pts=self.findChild(QLineEdit,"minpts")
        self.epsilon=self.findChild(QLineEdit,"eps")
        self.range1=self.findChild(QLineEdit,"range")




        self.btn1 = self.findChild(QPushButton,"btn1")
        self.btn2 = self.findChild(QPushButton,"btn2")
        self.btn3 = self.findChild(QPushButton,"btn3")
        self.btn4 = self.findChild(QPushButton,"btn4")


        self.hist_column=self.findChild(QComboBox,"hist_column")
        self.hist_column_add=self.findChild(QComboBox,"hist_column_add")
        self.hist_add_btn = self.findChild(QPushButton,"hist_add_btn")
        self.hist_remove_btn = self.findChild(QPushButton,"hist_remove_btn")
        self.histogram_btn = self.findChild(QPushButton,"histogram")

        self.heatmap_btn = self.findChild(QPushButton,"heatmap")



        self.Elbow_btn.clicked.connect(self.elbow)
        self.Elbow2_btn.clicked.connect(self.elbow2)

        self.btn1.clicked.connect(self.btn_1)
        self.btn2.clicked.connect(self.btn_2)
        self.btn3.clicked.connect(self.btn_3)
        self.btn4.clicked.connect(self.btn_4)



        self.columns.clicked.connect(self.target)
        self.Browse.clicked.connect(self.getCSV)
        self.Drop_btn.clicked.connect(self.dropc)
        self.kmean_btn.clicked.connect(self.kmeans_plot)
        self.kmedoid_btn.clicked.connect(self.kmedoid_plot)
        self.agnes_btn.clicked.connect(self.agnes_plot)
        self.diana_btn.clicked.connect(self.diana_plot)
        self.dbscan_btn.clicked.connect(self.dbscan_plot)
        self.perf_btn.clicked.connect(self.perf_plot)



        
        self.fillna_btn.clicked.connect(self.fillna)
        self.fillmean_btn.clicked.connect(self.fillme)
        
        self.hist_add_btn.clicked.connect(self.hist_add_column)
        self.hist_remove_btn.clicked.connect(self.hist_remove_column)
        self.histogram_btn.clicked.connect(self.plot_histogram)

        self.heatmap_btn.clicked.connect(self.heatmap_gen)

        self.con_btn.clicked.connect(self.con_cat)
        self.submit_btn.clicked.connect(self.set_target)

        self.train=self.findChild(QPushButton,"train")
#        self.train.clicked.connect(self.train_func)
        self.scale_btn.clicked.connect(self.scale_value)
        
 #       self.pre_trained.clicked.connect(self.upload_model)
  #      self.go_pre_trained.clicked.connect(self.test_pretrained)
        self.show()



    def elbow(self):
        data.elbow_(self.df)
    def elbow2(self):
        data.elbow_2(self.df)
           
    def btn_1(self):
        intra, inter = data.btn_1(df=self.df,k=self.scatter_c.text())
        self.output1.setText(str(intra))
        self.output11.setText(str(inter))
    def btn_2(self):
        intra, inter = data.btn_2(df=self.df,k=self.plot_c.text())
        self.output2.setText(str(intra))
        self.output22.setText(str(inter))
    def btn_3(self):
        intra, inter = data.btn_3(df=self.df,k=self.agnes_k.text())
        self.output3.setText(str(intra))
        self.output33.setText(str(inter))
    def btn_4(self):
        intra, inter = data.btn_4(df=self.df,k=self.diana_k.text())
        self.output4.setText(str(intra))
        self.output44.setText(str(inter))
    def scale_value(self):
        
        #my_dict={"StandardScaler":standard_scale ,"MinMaxScaler":min_max, "PowerScaler":power_scale}
        if self.scaler.currentText()=='StandardScale':
            self.df,func_name = data.StandardScale(self.df)
        elif self.scaler.currentText()=='MinMaxScale':
            self.df,func_name = data.MinMaxScale(self.df)
        elif self.scaler.currentText()=='PowerScale':
            self.df,func_name = data.PowerScale(self.df)
        
        steps.add_text(self.scaler.currentText()+" applied to data")
        steps.add_pipeline(self.scaler.currentText(),func_name)
        self.filldetails()


    def hist_add_column(self):

        self.hist_column_add.addItem(self.hist_column.currentText())
        self.hist_column.removeItem(self.hist_column.findText(self.hist_column.currentText()))


    def hist_remove_column(self):
        
        self.hist_column.addItem(self.hist_column_add.currentText())
        self.hist_column_add.removeItem(self.hist_column_add.findText(self.hist_column_add.currentText()))


        
        
    def heatmap_gen(self):

        data.plot_heatmap(self.df)

    def set_target(self):

        self.target_value=str(self.item.text()).split()[0]
        steps.add_code("target=data['"+self.target_value+"']")
        self.target_col.setText(self.target_value)

    def filldetails(self,flag=1):
         
        if(flag==0):  
            self.df = data.read_file(str(self.filePath))
        
        
        self.columns.clear()
        self.column_list=data.get_column_list(self.df)
        self.empty_list=data.get_empty_list(self.df)
        self.cat_col_list=data.get_cat(self.df)
        for i ,j in enumerate(self.column_list):
            stri=j+ " -------   " + str(self.df[j].dtype)
            self.columns.insertItem(i,stri)
            

        self.fill_combo_box() 
        shape_df="Shape:  Rows:"+ str(data.get_shape(self.df)[0])+"  Columns: "+str(data.get_shape(self.df)[1])
        self.data_shape.setText(shape_df)

    def fill_combo_box(self):
        
        self.dropcolumns.clear()
        self.dropcolumns.addItems(self.column_list)
        self.emptycolumn.clear()
        self.emptycolumn.addItems(self.empty_list)
        self.cat_column.clear()
        self.cat_column.addItems(self.cat_col_list)
        self.scatter_x.clear()
        self.scatter_x.addItems(self.column_list)
        self.scatter_y.clear()
        self.scatter_y.addItems(self.column_list)
        self.plot_x.clear()
        self.plot_x.addItems(self.column_list)
        self.plot_y.clear()
        self.plot_y.addItems(self.column_list)
        self.hist_column.clear()
        self.hist_column.addItems(data.get_numeric(self.df))
        self.hist_column.addItem("All")

        
        #self.describe.setText(data.get_describe(self.df))
        
        x=table_display.DataFrameModel(self.df)
        self.table.setModel(x)
        
    def upload_model(self):
        self.filePath_pre, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'Users/Alaa/Documents/GitHub/clustering-app/dataset')
        with open(self.filePath_pre, 'rb') as file:
            self.pickle_model = pickle.load(file)
        
    def test_pretrained(self):

        self.testing=pre_trained.UI(self.df,self.target_value,self.pickle_model,self.filePath_pre)

    def con_cat(self):
        
        a=self.cat_column.currentText()
        self.df[a],func_name =data.convert_category(self.df,a)
        steps.add_text("Column "+ a + " converted using LabelEncoder")
        steps.add_pipeline("LabelEncoder",func_name)
        self.filldetails()

    def fillna(self):

        self.df[self.emptycolumn.currentText()]=data.fillna(self.df,self.emptycolumn.currentText())
        code="data['"+self.emptycolumn.currentText()+"'].fillna('"'Uknown'"',inplace=True)"
        steps.add_code(code)
        steps.add_text("Empty values of "+ self.emptycolumn.currentText() + " filled with Uknown")
        self.filldetails()

    def fillme(self):

        self.df[self.emptycolumn.currentText()]=data.fillmean(self.df,self.emptycolumn.currentText())
        code="data['"+self.emptycolumn.currentText()+"'].fillna(data['"+self.emptycolumn.currentText()+"'].mean(),inplace=True)"
        steps.add_code(code)
        steps.add_text("Empty values of "+ self.emptycolumn.currentText() + " filled with mean value")
        self.filldetails()

    def getCSV(self):
        self.filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/akshay/Downloads/ML Github/datasets',"arff(*.arff)")
        self.columns.clear()
        
        code="pd.DataFrame((data=arff.loadarff('"+str(self.filePath)+"'))[0])"
        steps.add_code(code)
        steps.add_text("File "+self.filePath+" read")
        if(self.filePath!=""):
            self.filldetails(0)


    def target(self):
        self.item=self.columns.currentItem()
        
     
 
    def dropc(self):

        if (self.dropcolumns.currentText() == self.target_value):
            self.target_value=""
            self.target_col.setText("")
        self.df=data.drop_columns(self.df,self.dropcolumns.currentText())
        steps.add_code("data=data.drop('"+self.dropcolumns.currentText()+"',axis=1)")
        steps.add_text("Column "+ self.dropcolumns.currentText()+ " dropped")
        self.filldetails()  

    def kmeans_plot(self):

        data.kmeans_plot(df=self.df,x=self.scatter_x.currentText(),y=self.scatter_y.currentText(),k=self.scatter_c.text(),marker=self.scatter_mark.currentText())

        

    def kmedoid_plot(self):

        data.kmedoid_plot(df=self.df,x=self.plot_x.currentText(),y=self.plot_y.currentText(),k=self.plot_c.text(),marker=self.plot_mark.currentText())
     
    def agnes_plot(self):
        data.agnes_plot(df=self.df,k=self.agnes_k.text())
    def diana_plot(self):
        data.diana_plot(df=self.df,k=self.diana_k.text())
    def dbscan_plot(self):
        data.dbscan_plot(df=self.df,min_pts=self.minpts.text(),epsilon=self.eps.text(),range1=self.range1.text())
    def perf_plot(self):
        data.perf_plot(df=self.df,min_pts=self.minpts.text(),epsilon=self.eps.text(),range1=self.range1.text())

    def train_func(self):

        myDict={ "Linear Regression":linear_reg , "SVM":svm_model ,"SVR":SVR , "Logistic Regression":logistic_reg ,"Random Forest":RandomForest,
        "K-Nearest Neighbour":KNN ,"Multi Layer Perceptron":mlp ,"Gaussian NB":gaussian}
        
        if(self.target_value!=""):
            
            self.win = myDict[self.model_select.currentText()].UI(self.df,self.target_value,steps)
            
                    
 
    def histogram_plot(self):
        data.plot_histogram(self.df,k=self.scatter_c)
    def plot_histogram(self):
        intra1, inter1 = data.btn_1(df=self.df,k=self.hist_k.text())
        intra2, inter2 = data.btn_2(df=self.df,k=self.hist_k.text())
        intra3, inter3 = data.btn_3(df=self.df,k=self.hist_k.text())
        intra4, inter4 = data.btn_4(df=self.df,k=self.hist_k.text())

            
        # define the methods
        methods = ['K-Means', 'K-Medoids', 'AGNES', 'DIANA']

        # define the intra and inter measures for each method
        intra_measures = [intra1, intra2, intra3, intra4]
        inter_measures = [inter1, inter2, inter3, inter4]

        # plot the histogram for intra measures
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        ax[0].bar(methods, intra_measures)
        ax[0].set_title('Intra Measures')
        ax[0].set_xlabel('Methods')
        ax[0].set_ylabel('Values')

        # plot the histogram for inter measures
        ax[1].bar(methods, inter_measures)
        ax[1].set_title('Inter Measures')
        ax[1].set_xlabel('Methods')
        ax[1].set_ylabel('Values')

        plt.tight_layout()
        plt.show()
            


 
app = QApplication(sys.argv)
window = UI()
error_w=error_window()
app.exec_()