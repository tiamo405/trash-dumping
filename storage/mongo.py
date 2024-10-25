import os, sys
root = os.getcwd()
sys.path.append(root)
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from config import MONGO_URI
class MongoDBManager:
    def __init__(self, uri=MONGO_URI, db_name="doan"):
        self.client = MongoClient(uri)
        self.db = self.client.doan

    # Thêm tài liệu vào bộ sưu tập (insert vào camera)
    def insert_camera(self, cam_data):
        cameras = self.db['cameras']
        result = cameras.insert_one(cam_data)
        return str(result.inserted_id)

    # Thêm tài liệu vào bộ sưu tập tài khoản (insert vào account)
    def insert_account(self, account_data):
        accounts = self.db['customers']
        result = accounts.insert_one(account_data)
        return str(result.inserted_id)

    # Thêm role vào bộ sưu tập roles
    def insert_role(self, role_data):
        roles = self.db['roles']
        result = roles.insert_one(role_data)
        return str(result.inserted_id)
    
    # Thêm tài liệu vào bộ sưu tập violation_images (insert vào violation_images)
    def insert_violation(self, violation_data):
        violations = self.db['violation_images']
        result = violations.insert_one(violation_data)
        return str(result.inserted_id)
    # them video
    def insert_video(self, video_data):
        videos = self.db['videos']
        result = videos.insert_one(video_data)
        return str(result.inserted_id)

    # Tìm camera theo id
    def get_camera_by_id(self, cam_id):
        cameras = self.db['cameras']
        return cameras.find_one({"_id": ObjectId(cam_id)})

    # Tìm camera theo URL RTSP
    def get_camera_by_rtsp(self, rtsp_cam):
        cameras = self.db['cameras']
        return cameras.find_one({"rtsp_cam": rtsp_cam})

    # Tìm tài khoản theo username
    def get_account_by_username(self, username):
        accounts = self.db['customers']
        return accounts.find_one({"username": username})

    # Tìm role theo tên role
    def get_role_by_name(self, role_name):
        roles = self.db['roles']
        return roles.find_one({"role_name": role_name})
    

    # Cập nhật camera theo ID
    def update_camera_by_id(self, cam_id, update_data):
        cameras = self.db['cameras']
        result = cameras.update_one(
            {"_id": ObjectId(cam_id)},
            {"$set": update_data}
        )
        return result.modified_count
    
    # Cập nhật tài khoản theo ID
    def update_account_by_id(self, account_id, update_data):
        accounts = self.db['customers']
        result = accounts.update_one(
            {"_id": ObjectId(account_id)},
            {"$set": update_data}
        )
        return result.modified_count

    # Xóa camera theo ID
    def delete_camera_by_id(self, cam_id):
        cameras = self.db['cameras']
        result = cameras.delete_one({"_id": ObjectId(cam_id)})
        return result.deleted_count
    
    # Xóa tài khoản theo ID
    def delete_account_by_id(self, account_id):
        accounts = self.db['customers']
        result = accounts.delete_one({"_id": ObjectId(account_id)})
        return result.deleted_count
    
    # xoas role theo ID
    def delete_role_by_id(self, role_id):
        roles = self.db['roles']
        result = roles.delete_one({"_id": ObjectId(role_id)})
        return result.deleted_count

    # Lấy danh sách tất cả camera
    def get_all_cameras(self):
        cameras = self.db['cameras']
        return list(cameras.find())

    # Lấy danh sách camera đang mở
    def get_open_cameras(self):
        cameras = self.db['cameras']
        return list(cameras.find({"isOpen": True}))
    
    #lay danh sach tat ca tai khoan
    def get_all_accounts(self):
        accounts = self.db['customers']
        return list(accounts.find())

    #lay danh sach tat ca violation_images theo camera_id
    def get_all_violation_images_by_camId(self, camera_id):
        violations = self.db['violation_images']
        return list(violations.find({"camera_id": camera_id}))
    
    # Lấy danh sách tất cả violation_images theo camera_id, violation_date
    def get_all_violation_images(self, camera_id, violation_date):
        violations = self.db['violation_images']
        return list(violations.find({"camera_id": camera_id, "violation_date": violation_date}))
    
    # update camera
    def update_camera(self, camera):
        cameras = self.db['cameras']
        result = cameras.update_one(
            {"_id": camera['_id']},
            {"$set": camera}
        )
        return result.modified_count # 1 : ok, 0 : fail

# from pymongo.mongo_client import MongoClient as MongoClientAtlas    
# class MongoDBManagerAtlas:
#     def __init__(self, uri="mongodb+srv://nam05052002:<db_password>@cluster0.27uon.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"):
#         self.client = MongoClientAtlas(uri)
#         self.db = self.client['Cluster0']
    



# Ví dụ về việc sử dụng class này
if __name__ == "__main__":
    mongo_manager = MongoDBManager()

    new_camera = {
        "rtsp_cam": 'rtsp://cxview:gs252525@113.161.58.13:554/Streaming/Channels/701',
        "is_activate": True,
        "date_added": int(datetime.utcnow().timestamp()),
        "location": "video test 1",
        "add_by_customer_id": "671b0c1be9383de32aefd299",
    }
    # id = mongo_manager.insert_camera(new_camera)
    # print(id)
    # new_role = {
    #     "role": "ADMIN",
    #     "role_name": "admin",
    #     "permissions": ["create", "read", "update", "delete"],
    #     "date_create": int(datetime.utcnow().timestamp())
    # }
    # id = mongo_manager.insert_role(new_role)
    # print(id)

    new_account = {
        "username": "admin",
        "password": "admin",
        "email": "admin@gmail.com",
        "dob": "01/01/2000",
        "phoneNumber": "0123456789",
        "full_name": "tran nam",
        "date_create": datetime.utcnow().timestamp(),
        "role": str('671b0a215ebeec220d445bf7')
    }
    # id = mongo_manager.insert_account(new_account)
    # print(id)
    # find_camera = mongo_manager.get_camera_by_rtsp('test_model/video_test/2.mp4')
    # # find_camera['location'] = 'video test 1 2'
    # # modified = mongo_manager.update_camera(find_camera)
    # # find_camera = mongo_manager.get_camera_by_rtsp('test_model/video_test/2.mp4')
    # camId = str(find_camera['_id'])
    # # find image violation by id
    # find_violation = mongo_manager.get_all_violation_images(camId, 1729530000)
    # print(find_violation)

    # find_all_violation = mongo_manager.get_all_violation_images(cam_id, 1727715600)
    # print(find_all_violation)
