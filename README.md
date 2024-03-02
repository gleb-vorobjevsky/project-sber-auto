A project for the sberauto website that predicts the completion of a target action by the client.

The purpose of this project is to create a script that uploads data to a dataframe, processes data, and prepares them for training. Next, this script trains the model, saves it to a file. FastAPI is also implemented for local verification of the model's operability

since github does not allow you to download large amounts of data for free, the data for training the model can be downloaded here: https://drive.google.com/drive/folders/1yq8sb0PZXr42NXO-DJAAwwCxhrgXep9f?usp=sharing

The data for getting information contains the following data: 'session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', ' utm_keyword', 'device_category', 'device_os', 'device_brand', 'device_model', 'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city'

to test the model, you need to run a local server through git and connect using postman
