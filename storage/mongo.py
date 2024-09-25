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
        accounts = self.db['accounts']
        result = accounts.insert_one(account_data)
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
        accounts = self.db['accounts']
        return accounts.find_one({"username": username})

    # Cập nhật camera theo ID
    def update_camera_by_id(self, cam_id, update_data):
        cameras = self.db['cameras']
        result = cameras.update_one(
            {"_id": ObjectId(cam_id)},
            {"$set": update_data}
        )
        return result.modified_count

    # Xóa camera theo ID
    def delete_camera_by_id(self, cam_id):
        cameras = self.db['cameras']
        result = cameras.delete_one({"_id": ObjectId(cam_id)})
        return result.deleted_count

    # Lấy danh sách tất cả camera
    def get_all_cameras(self):
        cameras = self.db['cameras']
        return list(cameras.find())

    # Lấy danh sách camera đang mở
    def get_open_cameras(self):
        cameras = self.db['cameras']
        return list(cameras.find({"isOpen": True}))

    # Thêm role vào bộ sưu tập roles
    def insert_role(self, role_data):
        roles = self.db['roles']
        result = roles.insert_one(role_data)
        return str(result.inserted_id)

    # Tìm role theo tên role
    def get_role_by_name(self, role_name):
        roles = self.db['roles']
        return roles.find_one({"role_name": role_name})

# Ví dụ về việc sử dụng class này
if __name__ == "__main__":
    mongo_manager = MongoDBManager()

    # Thêm camera
    new_camera = {
        "rtsp_cam": "rtsp://192.168.1.101:554/stream1",
        "isOpen": True,
        "date_added": datetime.utcnow().timestamp(),
        "location": "Building B - Floor 2"
    }
    cam_id = mongo_manager.insert_camera(new_camera)
    print(f"Inserted camera with ID: {cam_id}")

    # Tìm camera theo RTSP URL
    # found_camera = mongo_manager.get_camera_by_rtsp("rtsp://192.168.1.101:554/stream1")
    # print(f"Found camera: {found_camera}")

    # # Cập nhật camera theo ID
    # update_data = {"isOpen": True}
    # updated_count = mongo_manager.update_camera_by_id(cam_id, update_data)
    # print(f"Number of cameras updated: {updated_count}")

    # # # Xóa camera theo ID
    
    # deleted_count = mongo_manager.delete_camera_by_id('66f37ceb37af25865bfb6e1e')
    # print(f"Number of cameras deleted: {deleted_count}")


