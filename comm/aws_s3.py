from boto3 import client
from face_recognition import load_image_file
import json
import os
from config import AWS_BUCKET_LOCATION


def get_client():
    return client(
        's3',
        AWS_BUCKET_LOCATION,
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY')
    )


def get_picture(imgpath, bucket):
    s3 = get_client()
    file = s3.get_object(Bucket=bucket, Key=imgpath)
    return file['Body']


def get_known_models_array(faces_messages, bucket):
    s3 = get_client()
    for face in faces_messages:
        file = s3.get_object(Bucket=bucket, Key=face["modelS3Path"])
        face["model"] = json.loads(file['Body'].read())
    return faces_messages


def get_files(bucket, faces_messages, imgpath):
    imagefile = get_picture(imgpath, bucket)
    image_narray = load_image_file(imagefile)
    known_faces_array = get_known_models_array(faces_messages, bucket)
    return image_narray, known_faces_array
