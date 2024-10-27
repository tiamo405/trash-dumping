# api
## get list cam
- curl
```sh
curl -X 'GET' \
  'http://127.0.0.1:5005/prod/api/v1/camera/list?page=1&limit=5' \
  -H 'accept: application/json'
```
- reponse:
```sh 
{
  "msg": "",
  "error_code": null,
  "code": 0,
  "entities": 
  {
    "cameras": [
      {
        "_id": "671b0ca3b236731f0056e606",
        "rtsp_cam": "rtsp://cxview:gs252525@113.161.58.13:554/Streaming/Channels/701",
        "is_activate": true,
        "date_added": 1729800755,
        "location": "GS HCM",
        "add_by_customer_id": "671b0c1be9383de32aefd299",
        "origin_image": "http://192.168.5.106:9000/trash-dumping/671b0ca3b236731f0056e606.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20241027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241027T032250Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=82116a738758c2ed945c0b36978dba55433ea689c6698e037c199ebdd901c24f"
      },
      {
        "_id": "671c65ba0b44007ee545bc1a",
        "rtsp_cam": "test_model/video_test/2.mp4",
        "is_activate": true,
        "date_added": 1729889098,
        "location": "video test 1",
        "add_by_customer_id": "671b0c1be9383de32aefd299",
        "origin_image": "http://192.168.5.106:9000/trash-dumping/671c65ba0b44007ee545bc1a.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20241027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241027T032250Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9ddabb21a9665642959095001567eb1d9001e0d95d66f4acb709bf4373759229"
      }
    ],
    "total_pages": 1
  }
}
``` 
## get history
- curl
```sh
curl -X 'GET' \
  'http://127.0.0.1:5005/prod/api/v1/view/get_history?rtsp_url=rtsp%3A%2F%2Fcxview%3Ags252525%40113.161.58.13%3A554%2FStreaming%2FChannels%2F701&date=1729875600&page=1&limit=3' \
  -H 'accept: application/json'
```
- reponse
```sh
{
  "msg": "",
  "error_code": null,
  "code": 200,
  "entities": 
  {
    "rtsp_url": "rtsp://cxview:gs252525@113.161.58.13:554/Streaming/Channels/701",
    "violations": [
      {
        "_id": "671c9e6ed3be9f0fbb249752",
        "camera_id": "671b0ca3b236731f0056e606",
        "detect_timestamp": 1729928814,
        "violation_date": 1729875600,
        "url_image": "http://192.168.5.106:9000/trash-dumping/671c9e6ed3be9f0fbb249752.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20241027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241027T032718Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=48fc2ee4db1c96967126a1f3926b259ee52b8a2839006b77582eb39ec5f184e8",
        "location": "GS HCM"
      },
      {
        "_id": "671c9f6f8556a7e23b91d114",
        "camera_id": "671b0ca3b236731f0056e606",
        "detect_timestamp": 1729929071,
        "violation_date": 1729875600,
        "url_image": "http://192.168.5.106:9000/trash-dumping/671c9f6f8556a7e23b91d114.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20241027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241027T032718Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c75c40361fb65ae2e5780685c4534c6d388f650d028563cc4e4db690d9770d4c",
        "location": "GS HCM"
      },
      {
        "_id": "671ca08a8556a7e23b91d115",
        "camera_id": "671b0ca3b236731f0056e606",
        "detect_timestamp": 1729929354,
        "violation_date": 1729875600,
        "url_image": "http://192.168.5.106:9000/trash-dumping/671ca08a8556a7e23b91d115.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20241027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241027T032718Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=075d9913ec403ce761496d218302cf6518d88cf1256dcd56d592e347bbb65000",
        "location": "GS HCM"
      }
    ],
    "total": 16
  }
}
```
## get video violation
- curl
```sh
curl -X 'GET' \
  'http://127.0.0.1:5005/prod/api/v1/view/get_history_video_violation?id_image_violation=671c9e6ed3be9f0fbb249752&page=1&limit=10' \
  -H 'accept: application/json'
```

- reponse
```sh
{
  "msg": "",
  "error_code": null,
  "code": 200,
  "entities": 
  {
    "videos": [
      {
        "_id": "671c9eda2e6edf0d0615dc4e",
        "camera_id": "671b0ca3b236731f0056e606",
        "video_path": "671b0ca3b236731f0056e606_1729928802.mp4",
        "start_time": 1729928802,
        "end_time": 1729928922,
        "date_time": 1729875600,
        "url_video": "http://192.168.5.106:9000/trash-dumping/671b0ca3b236731f0056e606_1729928802.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20241027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241027T032829Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=2966da323034fb6786996de8775644905f8ed96eace713a31ff219ace870319c"
      }
    ]
  }
}
```
