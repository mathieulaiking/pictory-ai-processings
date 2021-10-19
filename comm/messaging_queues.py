from ai.detection_pipeline import start_detections
from comm.aws_s3 import get_files
import os
import json
import pika
from config import CLOUD_QUEUE_ADDRESS


def message_processing(body):

    json_obj = json.loads(body)
    print("RECEIVED OBJECT : ", json_obj)
    image_dict = {"id": json_obj["id"]}
    # Getting files from s3 server
    image_narray, known_faces_array = get_files(
        json_obj["s3Bucket"], json_obj["faces"], json_obj["rawFilePath"])
    # Starting detections pipeline (faces,objects,blur,scenes)
    message = start_detections(image_narray, image_dict, known_faces_array)
    return message


def rabbitMQ_connect(queue_address):
    # Connection to RabbitMQ (on Cloud)
    url = os.environ.get('CLOUDAMQP_URL', queue_address)
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)
    return connection.channel()


def start_queue_listening():
    channel = rabbitMQ_connect(CLOUD_QUEUE_ADDRESS)
    # Declare reception queue
    channel.queue_declare(queue='pending_uploads', durable=True)

    def callback(ch, method, properties, body):
        message = message_processing(body)
        # Delivering message on another queue
        channel.queue_declare(queue='processed_uploads', durable=True)
        channel.basic_publish(exchange='', routing_key='processed_uploads',
                              body=json.dumps(message))
    channel.basic_consume(queue='pending_uploads',
                          on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()
