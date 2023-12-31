import torch
from torchvision import transforms
from PIL import Image 
import numpy as np
from flask import Flask, render_template,redirect, url_for,request,send_file, jsonify, send_from_directory
import boto3
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import botocore.exceptions
import zipfile
import shutil
from flask_mail import Mail, Message
import psycopg2
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import psutil
import subprocess
from datetime import datetime
import threading
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
# CORS(app, origins='http://13.228.104.12')
CORS(app)

GOOD = ['5AA', '4AA', '3AA', '5AB', '4AB']
FAIR = ['3AB', '6AA', '6AB', '5BA', '4BA', '3BA', '5BB', '4BB', '3BB', '6BA', '6BB', '2AA', '2AB', '2BA', '2BB']
BAD = ['5AC', '4AC', '3AC', '6AC', '5BC', '4BC', '3BC', '6BC', '2A-', '2BC', '2B-', '5CA', '4CA', '3CA', '6CA', '2CA', '2C-', '2-A', '5CB', '4CB', '3CB', '6CB', '2CB', '2-B', '5CC', '4CC', '3CC', '6CC', '2CC', '2-C', '2--', '1AA', '1AB', '1A-', '1BA', '1BB', '1B-', '1-A', '1-B', '1AC', '1CA', '1BC', '1CB', '1CC', '1C-','1-C', '1--']
fulllist = {}
map_location = torch.device('cpu')
for idx, i in enumerate(GOOD):
    fulllist[i] = 101.5 - 4 * (idx + 1.5)
for idx, i in enumerate(FAIR):
    fulllist[i] = 76 - float(25 / (len(FAIR) - idx))
for idx, i in enumerate(BAD):
    fulllist[i] = 51 - float(50 / (len(BAD) - idx))
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, 'Best_expansion.pth')
net = torch.load(path, map_location=torch.device('cpu'))

path = os.path.join(current_dir, 'Best_icm.pth')
net1 = torch.load(path, map_location=torch.device('cpu'))

path = os.path.join(current_dir, 'Best_te.pth')
net2 = torch.load(path, map_location=torch.device('cpu'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
net = net.to(device)
net1 = net1.to(device)
net2 = net2.to(device)
net.eval()
net1.eval()
net2.eval()


def transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])
    return transform


dict1 = {0: '-', 1: 'A', 2: 'B', 3: 'C'}
dict2 = {0: '-', 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


def process_image(file,clinic_name,patient_id):
    rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)
    img = Image.open(file)

    gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (512, 512))

    rgb_image[:, :, 0] = gray_image
    rgb_image[:, :, 1] = gray_image
    rgb_image[:, :, 2] = gray_image
    im2 = Image.fromarray(rgb_image)

    im2 = transform()(im2)
    im2 = im2[None, :, :, :]
    im2 = im2.to(device)
    torch.cuda.empty_cache()
    out = net(im2)
    out1 = net1(im2)
    out2 = net2(im2)
    del(im2)
    row = out.detach().cpu().numpy()
    row1 = out1.detach().cpu().numpy()
    row2 = out2.detach().cpu().numpy()
    del(out1)
    del(out2)
    del(out)
    newdat = np.argmax(row)
    newdat1 = np.argmax(row1)
    newdat2 = np.argmax(row2)
    fullgrade = str(dict2[newdat]) + str(dict1[newdat1]) + str(dict1[newdat2])
    fullgrade_value = str(dict2[newdat]) + str(dict1[newdat1]) + str(dict1[newdat2])
    if str(fullgrade) in GOOD:
        quality = 'Best quality embryo viable for Freezing'
    elif str(fullgrade) in FAIR:
        quality = 'Fair quality embryo, blastocyst developing'
    else:
        quality = 'Poor quality embryo'

    new_name = clinic_name +'_'+ patient_id +'_'+ str(file.filename)
    url = f"https://lab.genesysailabs.com/uploads/{new_name}"
    return {
        "class": quality,
        "filename": file.filename,
        "img": url,
        "percentage": str(fulllist[fullgrade])
    }

@app.route('/upload_process', methods=['POST'])
def upload_process():
    files = request.files.getlist('file')
    clinic_name = str(request.form.get('clinic_name'))
    patient_id = str(request.form.get('patient_id'))
    mainlist = []

    for file in files:
        mainlist.append(process_image(file,clinic_name,patient_id))

    return jsonify(mainlist)


# @app.route('/upload_aws', methods=['POST'])
# def upload_aws():
#     files = request.files.getlist('file')
#     clinic_name = str(request.form.get('clinic_name'))
#     patient_id = str(request.form.get('patient_id'))
#     for file in files:
#         new_name= clinic_name +'_'+ patient_id +'_'+file.filename
#         s3 = boto3.client('s3', aws_access_key_id='AKIAQRHO5VROCDJKKT5M', aws_secret_access_key='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')
#         s3.upload_fileobj(file,'out-genysis',new_name)
#     return "ok"

# @app.route('/upload_aws', methods=['POST'])
# def upload_aws():
#     return "Done"

UPLOAD_FOLDER = '/home/ubuntu/genesys/genesys/backend/temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_aws', methods=['POST'])
def upload_aws():
    files = request.files.getlist('file')
    clinic_name = request.form.get('clinic_name')
    patient_id = request.form.get('patient_id')
    print(clinic_name, patient_id)
    uploaded_files = []

    try:
        for file in files:
            if file.filename != '':
                filename = secure_filename(file.filename)
                new_name = clinic_name +'_'+ patient_id +'_'+ filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
                file.save(file_path)  # Save the file to the server's upload folder

                # Now upload the file to S3
                # s3 = boto3.client('s3', aws_access_key_id='AKIAQRHO5VROCDJKKT5M', aws_secret_access_key='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')
                # s3.upload_file(file_path, 'out-genysis', new_name)

                uploaded_files.append({
                    'filename': new_name,
                    'url': f'/uploads/{new_name}'  # URL to access the image on the server
                })
    except Exception as e:
        # If any error occurs during the process, handle it and respond with an error message
        return jsonify({'error': str(e)}), 500

    return jsonify(uploaded_files) 

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


    # for file in files:
    #     mainlist.append(embroyoClassifier(file,s3))
    #     file.save(os.path.join(current_dir, file.filename))
    #     # with open(os.path.join(current_dir, file.filename), "rb") as f:
    #         # s3.upload_fileobj(f, "BUCKET_NAME", "OBJECT_NAME")
    #     s3.upload_file(os.path.join(current_dir, file.filename),'out-genysis',file.filename)

# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Function to get system metrics
def get_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    flask_app_memory = get_service_memory('my-flask-app.service')
    db_memory = get_service_memory('db.service')
    flask_app_cpu = get_service_cpu('my-flask-app.service')
    db_cpu = get_service_cpu('db.service')
    return cpu_usage, memory_usage, disk_usage, flask_app_memory, db_memory, flask_app_cpu, db_cpu

# Function to get memory usage of a service
def get_service_memory(service_name):
    try:
        output = subprocess.check_output(['systemctl', 'show', '--value', '--property=MemoryCurrent', service_name])
        memory_usage_kb = int(output.decode('utf-8'))
        memory_usage_mb = memory_usage_kb / 1024 / 1024
        return memory_usage_mb
    except subprocess.CalledProcessError:
        return None

# Function to get CPU usage of a service
def get_service_cpu(service_name):
    try:
        process = subprocess.Popen(['ps', '-eo', 'pid,%cpu,cmd', '--sort=-%cpu'], stdout=subprocess.PIPE)
        output = process.communicate()[0]
        lines = output.decode('utf-8').split('\n')
        for line in lines:
            if service_name in line:
                cpu_usage = float(line.split()[1])
                return cpu_usage
        return None
    except subprocess.CalledProcessError:
        return None

# Schedule the metric collection every 5 minutes
# scheduler.add_job(get_system_metrics, 'interval', minutes=5)

@app.route('/')
def dashboard():
    cpu_usage, memory_usage, disk_usage, flask_app_memory, db_memory, flask_app_cpu, db_cpu = get_system_metrics()
    return render_template('dashboard.html', cpu_usage=cpu_usage, memory_usage=memory_usage,
                           disk_usage=disk_usage, flask_app_memory=flask_app_memory, db_memory=db_memory,
                           flask_app_cpu=flask_app_cpu, db_cpu=db_cpu)

# @app.route('/')
# def test():
#     return("worked")

# source_bucket_name = 'genysis'
s3 = boto3.client(
    's3',
    aws_access_key_id='AKIAQRHO5VROCDJKKT5M',
    aws_secret_access_key='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'
)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/Utilities')
def utilities():
    return render_template('utilities.html')

def get_total_objects(s3_client, bucket_name):
    total_objects = 0
    continuation_token = None

    while True:
        params = {'Bucket': bucket_name}
        if continuation_token:
            params['ContinuationToken'] = continuation_token

        response = s3_client.list_objects_v2(**params)
        objects = response.get('Contents', [])

        total_objects += len(objects)

        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    return total_objects

def total_objects(s3_client, bucket_name):
    all_objects=[]
    continuation_token = None

    while True:
        params = {'Bucket': bucket_name}
        if continuation_token:
            params['ContinuationToken'] = continuation_token

        response = s3_client.list_objects_v2(**params)
        objects = response.get('Contents', [])

        all_objects.extend(objects)

        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    return all_objects

@app.route('/Annotated-counts', methods=['GET', 'POST'])
def Annotated_counts():
    s3_client = boto3.client(
        's3',
        aws_access_key_id='AKIAQRHO5VROCDJKKT5M',
        aws_secret_access_key='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'
    )
    objects = total_objects(s3_client, 'embryologist-marked')   
    image_details = [{'name': obj['Key'], 'last_modified': obj['LastModified']} for obj in objects]
    selected_date_str = request.args.get('last_modified_date')

    if selected_date_str:
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
    else:
        selected_date = datetime.today().date()

    bucket_esther = 'bucketesther'
    bucket_papitha = 'bucketpapitha'
    bucket_usha = 'bucketsusha'
    bucket_swathi = 'bucketsswathi'
    bucket_ann = 'bucketsann'
    bucket_asha = 'bucketasha'
    bucket_john = 'bucketsjohn'
    bucket_himaja = 'buckethimaja'
    bucket_raksha = 'bucketsraksha'
    count_papitha = 0
    count_usha = 0
    count_esther = 0
    count_swathi = 0
    count_ann = 0
    count_asha = 0
    count_john = 0
    count_himaja = 0
    count_raksha = 0
    t_count_papitha = 0
    t_count_usha = 0
    t_count_esther = 0
    t_count_swathi = 0
    t_count_ann = 0
    t_count_asha = 0
    t_count_john = 0
    t_count_himaja = 0
    t_count_raksha = 0
    total_images = get_total_objects(s3_client, 'embryologist-marked')

    total_esther = get_total_objects(s3_client, bucket_esther)
    total_papitha = get_total_objects(s3_client, bucket_papitha)
    total_usha = get_total_objects(s3_client, bucket_usha)
    total_swathi = get_total_objects(s3_client, bucket_swathi)
    total_ann = get_total_objects(s3_client, bucket_ann)
    total_asha = get_total_objects(s3_client, bucket_asha)
    total_john = get_total_objects(s3_client, bucket_john)
    total_himaja = get_total_objects(s3_client, bucket_himaja)
    total_raksha = get_total_objects(s3_client, bucket_raksha)


    for image in image_details:
        name = image['name']
        last_modified_date = image['last_modified'].date()

        if name.startswith('Papitha'):
            t_count_papitha += 1
        elif name.startswith('Usha'):
            t_count_usha += 1
        elif name.startswith('Esther'):
            t_count_esther += 1
        elif name.startswith('Swathi'): 
            t_count_swathi += 1
        elif name.startswith('Ann'):
            t_count_ann += 1
        elif name.startswith('Asha'):
            t_count_asha += 1
        elif name.startswith('John'):
            t_count_john += 1
        elif name.startswith('Himaja'):
            t_count_himaja += 1
        elif name.startswith('Raksha'):
            t_count_raksha += 1

        if selected_date and selected_date == last_modified_date:
            if name.startswith('Papitha'):
                count_papitha += 1
            elif name.startswith('Usha'):
                count_usha += 1
            elif name.startswith('Esther'):
                count_esther += 1
            elif name.startswith('Swathi'): 
                count_swathi += 1
            elif name.startswith('Ann'):
                count_ann += 1
            elif name.startswith('Asha'):
                count_asha += 1
            elif name.startswith('John'):
                count_john += 1
            elif name.startswith('Himaja'):
                count_himaja += 1
            elif name.startswith('Raksha'):
                count_raksha += 1
        

    return render_template('image_count.html',
                           count_papitha=count_papitha,count_usha=count_usha,count_esther=count_esther,count_swathi=count_swathi,count_ann=count_ann,count_asha=count_asha,
                           count_john=count_john,count_himaja=count_himaja,count_raksha=count_raksha,
                           t_count_papitha=t_count_papitha,t_count_usha=t_count_usha,t_count_esther=t_count_esther,t_count_swathi=t_count_swathi,
                           t_count_ann=t_count_ann,t_count_asha=t_count_asha,t_count_john=t_count_john,t_count_himaja=t_count_himaja,t_count_raksha=t_count_raksha,
                           total_images=total_images,selected_date=selected_date,
                           total_esther=total_esther,total_papitha=total_papitha,total_usha=total_usha,total_swathi=total_swathi,
                           total_ann=total_ann,total_asha=total_asha,total_john=total_john,total_himaja=total_himaja,total_raksha=total_raksha)




@app.route('/Annotation')
def index():
    print(request)
    current_index = int(request.args.get('index', '0'))
    new_name = request.args.get('new_name')
    image_data = get_image_data(new_name)
    total_images = len(image_data)
    if current_index >= total_images:
        current_index = 0
    elif current_index < 0:
        current_index = total_images - 1
    return render_template('index.html', new_name=new_name,
                           current_image=image_data[current_index], current_index=current_index,total_images=total_images,image_data=image_data)

@app.route('/test')
def index_test():
    print(request)
    current_index = int(request.args.get('index', '0'))
    new_name = request.args.get('new_name')
    image_data = get_image_data(new_name)
    total_images = len(image_data)
    if current_index >= total_images:
        current_index = 0
    elif current_index < 0:
        current_index = total_images - 1
    return render_template('test.html', new_name=new_name,
                           current_image=image_data[current_index], current_index=current_index,total_images=total_images,image_data=image_data)


@app.route('/save-s3', methods=['POST'])
def save_s3():
    try:
        new_bucket_name = 'embryologist-marked'
        new_filename = request.form.get('new_filename')
        current_image_url = request.form.get('current_image_url')
        current_image_filename = request.form.get('current_image_filename')
        current_index =request.form.get('current_indexx')
        new_name = request.form.get('new_name')
        if new_name == 'Esther':
            source_bucket_name = 'bucketesther'
        elif new_name == 'Papitha':
            source_bucket_name = 'bucketpapitha'
        elif new_name == 'Usha':
            source_bucket_name = 'bucketsusha'
        elif new_name == 'Swathi':
            source_bucket_name = 'bucketsswathi'
        elif new_name == 'Ann':
            source_bucket_name = 'bucketsann'
        elif new_name == 'Asha':
            source_bucket_name = 'bucketasha'
        elif new_name == 'John':
            source_bucket_name = 'bucketsjohn'
        elif new_name == 'Himaja':
            source_bucket_name = 'buckethimaja'
        elif new_name == 'Raksha':
            source_bucket_name = 'bucketsraksha'
        s3.copy_object(
            Bucket=new_bucket_name,
            CopySource={'Bucket': source_bucket_name, 'Key': current_image_filename},
            Key=new_filename
        )
        current_index=int(current_index)
        next_index = current_index + 1
        return redirect(url_for('index', index=next_index,new_name=new_name))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            return f"The bucket '{new_bucket_name}' does not exist or is not accessible."
        else:
            return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"



def get_image_data(selected_name):
    image_data = []

    try:
        if selected_name == 'Esther':
            source_bucket_name = 'bucketesther'
        elif selected_name == 'Papitha':
            source_bucket_name = 'bucketpapitha'
        elif selected_name == 'Usha':
            source_bucket_name = 'bucketsusha'
        elif selected_name == 'Swathi':
            source_bucket_name = 'bucketsswathi'
        elif selected_name == 'Ann':
            source_bucket_name = 'bucketsann'
        elif selected_name == 'Asha':
            source_bucket_name = 'bucketasha'
        elif selected_name == 'John':
            source_bucket_name = 'bucketsjohn'
        elif selected_name == 'Himaja':
            source_bucket_name = 'buckethimaja'
        elif selected_name == 'Raksha':
            source_bucket_name = 'bucketsraksha'

        continuation_token = None

        while True:
            if continuation_token:
                objects = s3.list_objects_v2(Bucket=source_bucket_name, ContinuationToken=continuation_token)
            else:
                objects = s3.list_objects_v2(Bucket=source_bucket_name)

            for item in objects.get('Contents', []):
                if item['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    url = f"https://{source_bucket_name}.s3.amazonaws.com/{item['Key']}"
                    filename = item['Key'].split('/')[-1]
                    image_data.append({'url': url, 'filename': filename})

            if objects['IsTruncated']:
                continuation_token = objects['NextContinuationToken']
            else:
                break

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"The bucket '{source_bucket_name}' does not exist or is not accessible.")
        else:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

    return image_data

# bucket_name = 'embryologist-marked'
bucket_name = 'genysis'
local_dir = 'local_images/'

def zip_and_download(source_folder, zip_filename):
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        return f"Folder '{source_folder}' does not exist."

    # Create a temporary directory to store the zip file
    temp_dir = 'temp_zip'
    os.makedirs(temp_dir, exist_ok=True)

    # Zip the contents of the source folder
    zip_filepath = os.path.join(temp_dir, zip_filename)
    
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_folder)
                zipf.write(file_path, arcname=arcname)

    return send_file(zip_filepath, as_attachment=True, download_name=zip_filename)

def count_total_files():
    total_files = 0
    continuation_token = None

    while True:
        if continuation_token:
            objects = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
        else:
            objects = s3.list_objects_v2(Bucket=bucket_name)

        total_files += len(objects.get('Contents', []))

        if objects['IsTruncated']:
            continuation_token = objects['NextContinuationToken']
        else:
            break

    return total_files

total_files = count_total_files()

@app.route('/download_all_images', methods=['POST'])
def download_all_images():
    all_files = os.listdir(local_dir)
    for file_name in all_files:
        file_path = os.path.join(local_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    def download_images():
        continuation_token = None
        while True:
            if continuation_token:
                objects = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
            else:
                objects = s3.list_objects_v2(Bucket=bucket_name)

            for obj in objects.get('Contents', []):
                object_key = obj['Key']
                local_file_path = os.path.join(local_dir, os.path.basename(object_key))
                s3.download_file(bucket_name, object_key, local_file_path)
            zip_filename = 'local_images.zip'  # Replace with the desired zip filename
            source_folder = 'local_images/'
            zip_and_download(source_folder, zip_filename)
            if objects['IsTruncated']:
                continuation_token = objects['NextContinuationToken']
            else:
                break
    download_thread = threading.Thread(target=download_images)
    download_thread.start()
    return jsonify({"message": "Downloading "+ str(total_files) + "images"})

@app.route('/download_specific_image', methods=['POST'])
def download_specific_image():
    all_files = os.listdir(local_dir)
    for file_name in all_files:
        file_path = os.path.join(local_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    filename_prefix = request.form.get('filename_prefix')
    if filename_prefix:
        def download_images():
            continuation_token = None
            while True:
                if continuation_token:
                    objects = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
                else:
                    objects = s3.list_objects_v2(Bucket=bucket_name)

                for obj in objects.get('Contents', []):
                    object_key = obj['Key']
                    if object_key.startswith(filename_prefix):
                        local_file_path = os.path.join(local_dir, os.path.basename(object_key))
                        s3.download_file(bucket_name, object_key, local_file_path)
                if objects['IsTruncated']:
                    continuation_token = objects['NextContinuationToken']
                else:
                    break

        download_thread = threading.Thread(target=download_images)
        download_thread.start()
        return jsonify({"message": f"Downloading images with prefix '{filename_prefix}'..."})
    else:
        return jsonify({"error": "Filename prefix not provided."})

@app.route('/delete_specific_image', methods=['POST'])
def delete_specific_image():
    filename_prefix = request.form.get('del_filename_prefix')
    if filename_prefix:
        continuation_token = None
        while True:
            if continuation_token:
                objects = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
            else:
                objects = s3.list_objects_v2(Bucket=bucket_name)

            for obj in objects.get('Contents', []):
                object_key = obj['Key']
                if object_key.startswith(filename_prefix):
                    s3.delete_object(Bucket=bucket_name, Key=object_key)
            if objects['IsTruncated']:
                continuation_token = objects['NextContinuationToken']
            else:
                break

        return jsonify({"message": f"Deleted images with prefix '{filename_prefix}'."})
    else:
        return jsonify({"error": "Filename prefix not provided."})

@app.route('/delete_all_images', methods=['POST'])
def delete_all_images():
    continuation_token = None
    while True:
        if continuation_token:
            objects = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=continuation_token)
        else:
            objects = s3.list_objects_v2(Bucket=bucket_name)

        for obj in objects.get('Contents', []):
            object_key = obj['Key']
            s3.delete_object(Bucket=bucket_name, Key=object_key)
        if objects['IsTruncated']:
            continuation_token = objects['NextContinuationToken']
        else:
            break

    return jsonify({"message": "Deleted all images."})


        

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',threaded=True,port=5000)
