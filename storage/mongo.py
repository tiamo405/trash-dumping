from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime

class MongoDBManager:
    def __init__(self, uri="mongodb://admin:admin@localhost:27017/", db_name="trash_dumping"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

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
    
    # Thêm tài liệu vào bộ sưu tập violation_videos (insert vào violation_videos)
    def insert_violation_video(self, violation_video_data):
        violation_videos = self.db['violation_videos']
        result = violation_videos.insert_one(violation_video_data)
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
    
    # Tìm violation_image theo id
    def get_violation_image_by_id(self, violation_id):
        violations = self.db['violation_images']
        return violations.find_one({"_id": ObjectId(violation_id)})
    
    # Tìm violation_video theo violation_image_id
    def get_violation_video_by_violation_id(self, violation_image_id):
        violation_videos = self.db['violation_videos']
        return violation_videos.find_one({"violation_image_id": violation_image_id})

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

    

# Ví dụ về việc sử dụng class này
if __name__ == "__main__":
    mongo_manager = MongoDBManager()

    # new_camera = {
    #     "rtsp_cam": 'test_model/video_test/2.mp4',
    #     "is_activate": True,
    #     "date_added": datetime.utcnow().timestamp(),
    #     "location": "video test",
    #     "add_by_customer_id": "66f50f9d6a42a68067ae030b",
    # }

    # new_role = {
    #     "role": "ADMIN",
    #     "role_name": "admin",
    #     "permissions": ["create", "read", "update", "delete"],
    #     "date_create": int(datetime.utcnow().timestamp())
    # }

    # new_account = {
    #     "username": "admin",
    #     "password": "admin",
    #     "email": "admin@gmail.com",
    #     "dob": "01/01/2000",
    #     "phoneNumber": "0123456789",
    #     "full_name": "tran nam",
    #     "date_create": datetime.utcnow().timestamp(),
    #     "role": str(role_id)
    # }

    find_camera = mongo_manager.get_camera_by_rtsp('test_model/video_test/2.mp4')
    # find_camera['location'] = 'video test 1 2'
    # modified = mongo_manager.update_camera(find_camera)
    # find_camera = mongo_manager.get_camera_by_rtsp('test_model/video_test/2.mp4')
    camId = str(find_camera['_id'])
    # find image violation by id
    find_violation = mongo_manager.get_all_violation_images(camId, 1729530000)
    print(find_violation)

    # find_all_violation = mongo_manager.get_all_violation_images(cam_id, 1727715600)
    # print(find_all_violation)
