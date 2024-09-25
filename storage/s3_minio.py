# file_uploader.py MinIO Python SDK example
import sys, os
root = os.getcwd()
sys.path.insert(0, root)

from minio import Minio
from minio.error import S3Error
from config import ENDPOINT, ACCESS_KEY, SECRET_KEY, SECURE, BUCKET

class S3Minio:
    def __init__(self, endpoint=ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=SECURE):
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = BUCKET

    def geturl(self, object_name):
        try:
            return self.client.presigned_get_object(self.bucket_name, object_name)
        except S3Error as e:
            print(e)

    def upload_file(self, file_path, object_name):
        try:
            self.client.fput_object(self.bucket_name, object_name, file_path)
        except S3Error as e:
            print(e)

    def download_file(self, object_name, file_path):
        try:
            self.client.fget_object(self.bucket_name, object_name, file_path)
        except S3Error as e:
            print(e)

    def list_files(self, collection_name):
        try:
            return self.client.list_objects(self.bucket_name, prefix=collection_name)
        except S3Error as e:
            print(e)

    def delete_file(self, object_name):
        try:
            self.client.remove_object(self.bucket_name, object_name)
        except S3Error as e:
            print(e)

    def create_bucket(self, bucket_name):
        try:
            self.client.make_bucket(bucket_name)
        except S3Error as e:
            print(e)

    def delete_bucket(self, bucket_name):
        try:
            self.client.remove_bucket(bucket_name)
        except S3Error as e:
            print(e)

if __name__ == "__main__":
    s3 = S3Minio()
    s3.upload_file(file_path='detection.avi', object_name='detection.avi')
    print(s3.geturl('detection.avi'))