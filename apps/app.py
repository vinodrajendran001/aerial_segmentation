import streamlit as st 
from streamlit.ReportThread import add_report_ctx
import SessionState
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
import subprocess
import threading
from glob import glob
from random import randint



def generate_json(mode,batch_size,shuffle,num_epochs,num_workers,learning_rate,learning_rate_decay,image_full_size_lst,split,train_percentage,image_resize,overlay,cities_lst,urls_lst):
        data_save_state = st.text('Generating json...')

        # creating json and adding the engine dtls 
        data_json = {}
        session_state.json_path =  ROOT_DIR
        filename_json = "config" + ".json"
        filepath_json = os.path.join(session_state.json_path, filename_json)

        if os.path.isfile(filepath_json):
                with open(filepath_json) as outfile:
                        Dict = json.load(outfile) 
        else:
                with open(filepath_json, "w") as outfile: 
                        json.dump(data_json, outfile, indent=4, separators=(',', ': ')) 
                with open(filepath_json) as outfile:
                        Dict = json.load(outfile)

        # with open(filepath_json) as outfile:
        #         Dict = json.load(outfile)    

        if overlay == "true":
                overlay = bool(overlay)
        else:
                overlay = bool("")  

        dataset = {
                "image_size_full" : image_full_size_lst,
                "split": split,
                "train_percentage": train_percentage,
                "image_resize": image_resize,
                "overlay": overlay,
                "city" : cities_lst,
                "url" :  urls_lst,              
        }

        # add data dtls
        data_dict = {'data': dataset}
        Dict.update(data_dict)


        model = {
                "arch": "net_Unet"
        }
        # add model dtls
        model_dict = {'model': model}
        Dict.update(model_dict)

        # add path of the data to the cycle

        if shuffle == "true":
                shuffle = bool(shuffle)
        else:
                shuffle = bool("")

        train = {
                "mode" : mode,                   
                "batch_size" :  batch_size,                  
                "shuffle" : shuffle,               
                "num_epochs" : num_epochs,                
                "num_workers" : num_workers,                
                "learning_rate" : learning_rate,           
                "learning_rate_decay" : learning_rate_decay,          
                "out_dir" : ""                
        }

        train_dict = {'train': train}
        Dict.update(train_dict)


        with open(filepath_json, 'w') as outfile:
                json.dump(Dict,outfile, indent=4, separators=(',', ': '))

        # Notifying the reader that the json was successfully generated.
        data_save_state.text('Generating json data...done!')

        return filepath_json   



# support running python scripts in background
def PopenCall(onExit, PopenArgs):
        def runInThread(onExit, PopenArgs):
                script_ID = PopenArgs[1]
                proc = subprocess.Popen(PopenArgs)
                with st.spinner("Running in background...."):
                        proc.wait()
                onExit(script_ID)
                return
    
        # thread creation code:
        thread = threading.Thread(target=runInThread, args=(onExit, PopenArgs))
        add_report_ctx(thread)
        thread.start()
        thread.join()
        
        return thread



def onExit(script_ID):
        # Do this function when the subprocess is finished
        st.success("Done processing" )


def initSession():

        sessionstate_init = SessionState.get(


                mode = "train",
                batch_size = 5,
                shuffle = "true",
                num_epochs = 10,
                num_workers = 5,
                learning_rate = 0.0001,
                learning_rate_decay = 0.00004,
                image_full_size = "2208, 2208",
                image_full_size_lst = [2208, 2208],
                split = 4,
                train_percentage = 80,
                image_resize = 300,
                overlay = "false",
                cities = "berlin",
                cities_lst = ["berlin"],
                urls = "https://zenodo.org/record/1154821/files/berlin.zip",
                urls_lst = ["https://zenodo.org/record/1154821/files/berlin.zip"],
                json_path = ""
        )

        return sessionstate_init


if __name__ == "__main__":


        # maintain session variables
        session_state = initSession()

        # set the datetime
        dt_now = datetime.now().strftime("%Y%m%d")

        # This is your Project Root
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # PAGES
        PAGES = [
                "Data",
                "Training",
                "Prediction",
        ]

        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", PAGES)

        st.title("Aerial image segmentation")

        #Data
        if selection == "Data":
                st.subheader("Download and process the data")
                session_state.image_full_size = st.text_input("Image full size", value=session_state.image_full_size)
                session_state.image_full_size_lst = [int(s) for s in session_state.image_full_size.split(',')]
                session_state.split = st.number_input("Split (#grids)", value=session_state.split)
                session_state.train_percentage = st.number_input("Train set percentage", value=session_state.train_percentage)                
                session_state.image_resize = st.number_input("Image resize", value=session_state.image_resize) 
                session_state.overlay = st.radio("Overlay", ('true', 'false'), index=1)
                session_state.cities = st.text_input("Cities", value=session_state.cities)
                session_state.cities_lst = [s for s in session_state.cities.split(',')]
                session_state.urls = st.text_area("urls", value=session_state.urls)
                session_state.urls_lst = [s for s in session_state.urls.split(',')]


                if st.button("Update"):
                        session_state.json_path = generate_json( session_state.mode,session_state.batch_size,session_state.shuffle,session_state.num_epochs,session_state.num_workers,session_state.learning_rate,session_state.learning_rate_decay,session_state.image_full_size_lst,session_state.split,session_state.train_percentage,session_state.image_resize,session_state.overlay,session_state.cities_lst,session_state.urls_lst)

                st.subheader("Download and process the dataset")
                if st.button("Download"):

                        if not session_state.json_path:
                                st.error("Json file is not generated")
                        else:
                                PopenArgs = [
                                "python", os.path.join("src", "data" , "make_dataset.py")
                                ]
                                st.spinner("Running {} in background.......".format(PopenArgs))
                                PopenCall(onExit, PopenArgs)
                                st.success("Downloaded dataset is saved in data/raw/")
                                st.success("Processed dataset is saved in data/interim/")  

                st.subheader("Prepare the training and test set")
                if st.button("Generate Train-Test"):
        
                        if not session_state.json_path:
                                st.error("Json file is not generated")
                        else:
                                PopenArgs = [
                                "python", os.path.join("src", "data" , "prepare_dataset.py")
                                ]
                                PopenCall(onExit, PopenArgs)
                                st.success("Train and Test dataset are saved in data/processed/")                        

        elif selection == "Training":
                st.subheader("Train and build the model")
                if st.checkbox("Define train parameters"):
                        session_state.mode = st.sidebar.selectbox('mode', ('train', 'test'), index=0)
                        session_state.batch_size = st.sidebar.number_input("Batch size", value=session_state.batch_size)    
                        session_state.shuffle = st.sidebar.selectbox('shuffle', ('true', 'false'), index=0)                    
                        session_state.num_epochs = st.sidebar.number_input("Epochs", value=session_state.num_epochs)  
                        session_state.num_workers = st.sidebar.number_input("Workers", value=session_state.num_workers)  
                        session_state.learning_rate = st.sidebar.number_input("Learning rate",format="%f", value=session_state.learning_rate)  
                        session_state.learning_rate_decay = st.sidebar.number_input("Learning rate decay",format="%f", value=session_state.learning_rate_decay)
                        if st.sidebar.button("Update"):
                                session_state.json_path = generate_json(session_state.mode,session_state.batch_size,
                                session_state.shuffle,session_state.num_epochs,session_state.num_workers,session_state.learning_rate,session_state.learning_rate_decay,session_state.image_full_size_lst,session_state.split,session_state.train_percentage,session_state.image_resize,session_state.overlay,session_state.cities_lst,session_state.urls_lst)
                

                st.subheader("Ready to run the training?")
                if st.button("Run Training"):
    
                        if not session_state.json_path:
                                st.error("Json file is not generated")
                        else:
                                PopenArgs = [
                                "python", os.path.join("src", "models" , "train_model.py")
                                ]
                                PopenCall(onExit, PopenArgs)
                                st.success("Mode file is saved in models/<latest datetimestamp>/")
                
                st.subheader("Ready to run the evaluation?")
                if st.button("Run Evaluation"):
        
                        if not session_state.json_path:
                                st.error("Json file is not generated")
                        else:
                                PopenArgs = [
                                "python", os.path.join("src", "models" , "evaluate_model.py")
                                ]
                                PopenCall(onExit, PopenArgs)
                                st.success("Evaluation results are saved in reports/<latest datetimestamp>/")



        elif selection == "Prediction":
                st.subheader("Prediction")
                model_checkpoint = st.text_input("Checkpoint file with complete path")
                st.info("Example: models/20200415-155601/trained_model_end.pth")
                image_path = st.text_input("Image file with complete path")   
                st.info("Example: data/raw/berlin/berlin5_image.png")
                out_file = st.text_input("Predicted mask file with complete path where you want to save")  
                st.info("Example: reports/figures/predict_berlin5_mask.png")          


                if st.button("Run Prediction"):
            
                        if not session_state.json_path:
                                st.error("Json file is not generated")
                        else:
                                PopenArgs = [
                                "python", os.path.join("src", "models" , "predict_model.py"), model_checkpoint, image_path, out_file
                                ]
                                PopenCall(onExit, PopenArgs)
                                st.success("Copy of predicted mask is saved in {}".format(out_file))
                                time.sleep(5)   

                if st.button("View the predicted mask"):
                        input_image = image_path
                        st.image(input_image, use_column_width=True, caption="Input Image")
                        pred_mask = out_file
                        st.image(pred_mask, use_column_width=True, caption="Predicted Mask")



